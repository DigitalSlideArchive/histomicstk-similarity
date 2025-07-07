import argparse
import json
import os
import pprint
import sys
import time
from typing import TypedDict, cast

import numpy as np
import tqdm
from slicer_cli_web import CLIArgumentParser

from histomicstk_similarity.EmbeddingSimilarity import models


class EmbeddingDict(TypedDict):
    tilesize: float
    stride: float
    model: str
    data: np.ndarray


def collect_batch(
        model: models.EmbeddingModel,
        batch: list[np.ndarray],
        batchcoor: list[tuple[int, int]] | list[list[int]],
        out: np.ndarray | None,
        maxx: int = 0,
        maxy: int = 0) -> np.ndarray | None:
    """
    Given a batch of image patches, run inference on them to get the embedding
    vectors and add them to our output.

    :param model: An embedding model that has been prepared and can run
        model.infer(batch) and return an array of embeddings.
    :param batch: a list of numpy image patches.
    :param batchcoor: a list of (x, y) index coordinates within the output of
        where to add the inference results.  These do not need to be distinct
        (for instance, multiple image matches all contribute to the same spot).
    :param out: a numpy array to store the embeddings.  If none, a new array
        is allocated that is (maxy, maxx, size of first embedding) shape and
        dtype of the first embedding.
    :param maxx: width of the output array if it needs to be allocated.
    :param maxy: height of the output array if it needs to be allocated.
    :returns: the output numpy array.  This will be the same as the input value
        if the input value was provided.
    """
    if len(batch) == 0:
        return out
    embeds = model.infer(batch)
    for idx in range(len(batch)):
        x, y = batchcoor[idx]
        embed = embeds[idx]
        if out is None:
            out = np.zeros((maxy, maxx, embed.shape[0]), dtype=embed.dtype)
        out[y, x, :] += embed
    return out


def generate_embedding(args: argparse.Namespace) -> EmbeddingDict:
    """
    Generate embeddings on an image.

    :param args: A namespace object with at least the following parameters.
        :param args.model: the name of the model to use.  This plus the suffix
            "Model" must match the name of a models.EmbeddingModel class.
        :type args.model: str
        :param args.image: the path to the image file to process.
        :type args.image: str | pathlib.Path
        :param args.tilesize: the tile size to use when computing embeddings.
            If this is less than the model's patch size, the model's match size
            will be used.  If greater than the model's patch size, multiple
            patches will have their embeddings averaged to produce a tile
            embedding.  If the tilesize is not an exact multiple of the model
            patch size, these patches will overlap.
        :type args.tilesize: int
        :param args.stride: the spacing between tiles.  A stride of 1 would
            compute embeddings centered on every pixel in the source image
            (except close to the edges).  If a stride larger than the tile size
            is specified, the tile size is used.
        :type args.stride: int
        :param args.batch: batch size for inference.  This will affect how much
            base and GPU memory is used.
        :type args.batch: int
        :param args.scale_um: image scale in microns used for inference.  If 0,
            the model's preferred scale is used.  If the source image does not
            include scale metadata or this is unspecified and a non-zero
            magnification is given, magnification is used.
        :type args.scale_um: float
        :param args.magnification: image magnification level used for
            inference.  If 0, the model's preferred magnification is used.  If
            the source image does not include magnification metadata, it is
            assumed to be "20".
        :type args.magnification: float
        :param args.embedout: an optional name of a numpy save file to store
            the results.  If the name doesn't end with an extension, numpy will
            automatically add .npz.
        :type args.embedout: str | pathlib.Path | None
    :returns: a dictionary of the embedding results.
    """
    import large_image
    import torch

    model = getattr(models, f'{args.model}Model')()
    print(f'Model {args.model} ({model.model_name}) prepared')
    ts = large_image.open(args.image)
    if args.tilesize < model.patch:
        args.tilesize = model.patch
    if args.stride > args.tilesize or args.stride < 1:
        args.stride = args.tilesize
    numsub = (args.tilesize + model.patch - 1) // model.patch
    substep = model.patch - ((model.patch * numsub - args.tilesize) // (
        numsub - 1)) if numsub > 1 else model.patch
    out = None
    batch = []
    batchcoor: list[tuple[int, int]] = []
    if args.batch < 1:
        args.batch = 1
    um = model.scale_um if args.scale_um <= 0 else args.scale_um
    if ts.metadata.get('mm_x') and um:
        scale = (um * 0.001) / ts.metadata['mm_x']
        scaleParam = {'scale': {'mm_x': um * 0.001, 'mm_y': um * 0.001}}
    else:
        mag = model.magnification if args.magnification <= 0 else args.magnification
        scale = (ts.metadata['magnification'] or 20) / mag
        scaleParam = {'scale': {'magnification': mag}}
        if not ts.metadata['magnification'] or ts.metadata['magnification'] < 1.5:
            # Assume the metadata is wrong and that we are actually a 20x scale
            # image
            scale = 20. / mag
            scaleParam = {'output': {'maxWidth': round(ts.sizeX * scale),
                                     'maxHeight': round(ts.sizeY * scale)}}
    with torch.no_grad(), torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
        for tile in tqdm.tqdm(ts.tileIterator(
            tile_size={'width': args.tilesize, 'height': args.tilesize},
            tile_overlap={'x': args.tilesize - args.stride, 'y': args.tilesize - args.stride},
            resample=True,
            format=large_image.constants.TILE_FORMAT_NUMPY,
            **scaleParam,
        ), mininterval=1 if os.isatty(sys.stdout.fileno()) else 30):
            if tile['width'] < args.tilesize or tile['height'] < args.tilesize:
                continue
            for dj in range(numsub):
                for di in range(numsub):
                    subimg = tile['tile'][
                        dj * substep: dj * substep + model.patch,
                        di * substep: di * substep + model.patch, :]
                    if subimg.shape[2] == 1:
                        subimg = np.repeat(subimg, 3, axis=2)
                    batch.append(subimg)
                    batchcoor.append((tile['level_x'], tile['level_y']))
                    if len(batch) == args.batch or out is None:
                        out = collect_batch(
                            model, batch, batchcoor, out,
                            tile['iterator_range']['level_x_max'],
                            tile['iterator_range']['level_y_max'])
                        batch = []
                        batchcoor = []
        out = collect_batch(model, batch, batchcoor, out)
    out /= numsub ** 2
    results: EmbeddingDict = {
        'tilesize': args.tilesize * scale,
        'stride': args.stride * scale,
        'model': args.model,
        'data': out,
    }
    if args.embedout:
        np.savez(args.embedout, **results)
    return results


def find_focus(
        keypoint: tuple[int, int] | list[int],
        embeds: EmbeddingDict,
        agg: int) -> tuple[int, int, float, float]:
    """
    Find the closest center of a calculated embedding value to a key point.
    Away from the edges, this will be no further than half the stride distance
    from the key point in either direction.

    :param keypoint: The x, y point in image pixel coordinates.
    :param embeds: the computed embedding values, along with the stride and
        tilesize.
    :param agg: The aggregation factor to use.  0 uses individually calculated
        embeddings.  Otherwise, (agg + 1) x (agg + 1) embeddings are averged
        together.
    :returns: the index offset values in x, y, and the pixel coordinates in x,
        y of the key point that will actually be used.
    """
    tx = ty = -1
    ti = tj = 0
    for j in range(embeds['data'].shape[0] - agg):
        y = int((j + agg * 0.5) * embeds['stride'] + embeds['tilesize'] // 2)
        for i in range(embeds['data'].shape[1] - agg):
            x = int((i + agg * 0.5) * embeds['stride'] + embeds['tilesize'] // 2)
            if (((keypoint[0] - x) ** 2 + (keypoint[1] - y) ** 2) <
                    ((keypoint[0] - tx) ** 2 + (keypoint[1] - ty) ** 2)):
                tx, ty = x, y
                ti, tj = i, j
    return ti, tj, tx, ty


def calculate_similarity(embeds: EmbeddingDict, args: argparse.Namespace) -> None:
    """
    Calculate similarity across an image and generate a heatmap based on a set
    of pre-calculated feature embeddings.

    :param embeds: a precalculated set of embeddings.
    :param args: A namespace object with at least the following parameters.
        :param args.aggregate: The aggregation factor to use.  0 uses
            individually calculated embeddings.  Otherwise, (agg + 1) x
            (agg + 1) embeddings are averged together.
        :type args.aggregate: int
        :param args.keypoint: The x, y point in image pixel coordinates to use
            as the basis for comparison.
        :type args.keypoint: tuple[int, int] | list[int]
        :param args.threshold: similarity is measured on a scale of -1 to 1.
            Values below this threshold will be treated as completely
            dissimilar and not appear in the heatmap.  Values between the
            threshold and 1 are scale from transparent to the full heatmap
            color.
        :type args.threshold: float
        :param args.color: the color of the heatmap.  This is a css color
            string (e.g, "#FF0000", "rgba(255, 0, 0, 0.8)", etc.).
        :type args.color: str
        :param args.girderApiUrl: If specified, the heatmap will be uploaded
            as an annotation to this girder instance.
        :type args.girderApiUrl: str | None
        :param args.girderToken: if specified, this is used to authenticate
            any uploads to girder.
        :type args.girderToken: str | None
        :param args.annotationid: if specified, a girder annotation id.  If
            uploading to girder, this annotation will be updated (replaced)
            with the new heatmap.
        :type args.annotationid: str | None
        :param args.imageid: if specified, a girder image item id.  If
            uploading to girder and an annotationid is not specified, the
            heatmap annotation will be added to this image.
        :type args.imageid: str | None
    """
    start = time.time()
    agg = max(0, args.aggregate - 1)
    ti, tj, tx, ty = find_focus(args.keypoint, embeds, agg)
    print(f'Using {tx}, {ty} as focus (asked for {args.keypoint[0]}, {args.keypoint[1]})')
    print(f'Using {ti}, {tj} as comparison column, row')
    data = embeds['data'].astype(float)
    if agg:
        data = np.lib.stride_tricks.sliding_window_view(
            data, window_shape=(agg + 1, agg + 1, 1))[..., 0].mean(axis=(3, 4))
    norms = np.linalg.norm(data, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    data = data / norms
    match = data[tj][ti]
    print(f'Normalized: {time.time() - start:5.3f}s elapsed')
    similarity = np.sum(data * match, axis=-1)
    similarity = np.clip((similarity - args.threshold) / (1 - args.threshold), 0, 1)
    print(f'Similarity: {time.time() - start:5.3f}s elapsed')
    points: list[list[float]] = []
    for j in range(similarity.shape[0]):
        y = int((j + agg * 0.5) * embeds['stride'] + embeds['tilesize'] // 2)
        for i in range(similarity.shape[1]):
            x = int((i + agg * 0.5) * embeds['stride'] + embeds['tilesize'] // 2)
            val = similarity[j][i]
            if i == ti and j == tj:
                print(f'Heatmap value at comparison {x}, {y}: {val}')
            if val > 0:
                points.append([x, y, 0, float(val)])
    print(f'{len(points)} total points')
    heatmap = {
        'type': 'heatmap',
        'points': points,
        'radius': embeds['stride'] * 2.5,
        'rangeValues': [0, 1],
        'scaleWithZoom': True,
        'colorRange': ['rgba(0, 0, 0, 0)', args.color.strip() or '#FFFF00'],
    }
    annot = {
        'name': f'Heatmap {tx}, {ty}',
        'elements': [heatmap],
    }
    print(f'Heatmap: {time.time() - start:5.3f}s elapsed')
    if args.girderApiUrl:
        import girder_client

        gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
        gc.token = args.girderToken
        if args.annotationid:
            gc.put(f'annotation/{args.annotationid.strip()}', data=json.dumps(annot))
        else:
            try:
                itemId = gc.get(f'file/{args.imageid}')['itemId']
            except Exception:
                itemId = gc.get(f'item/{args.imageid}')['_id']
            gc.post(f'annotation/item/{itemId}', data=json.dumps(annot))
    else:
        print(json.dumps(heatmap))
    print(f'Done: {time.time() - start:5.3f}s elapsed')


def main(args):
    print('\n>> CLI Parameters ...\n')
    pprint.pprint(vars(args))

    start = time.time()
    if args.image and not args.embedin:
        embeds = generate_embedding(args)
        print(f'Generation time: {time.time() - start:5.3f}s')
    else:
        loaded = np.load(args.embedin, allow_pickle=True)
        embeds = cast(EmbeddingDict, {
            k: loaded[k].item() if loaded[k].ndim == 0 else loaded[k] for k in loaded.files})
        print(f'Load time: {time.time() - start:5.3f}s')
    print(f'Data shape {embeds["data"].shape}')
    if args.keypoint[0] != -1 or args.keypoint[1] != -1:
        calculate_similarity(embeds, args)
    # If we are in an isolated girder job, don't output the input file
    if args.imageid and not args.image or args.embedin and args.embedin == args.embedout:
        os.unlink(args.embedout)
    return True


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
