# Notes: use lazy imports for heavy dependencies, like torch or timm.

import abc

import numpy as np


class EmbeddingModelMeta(abc.ABCMeta):
    """
    Add a metaclass to ensure our subclasses define all necessary attributes.
    """

    def __new__(cls, name, bases, attrs):
        if name != 'EmbeddingModel' and 'model_name' not in attrs:
            msg = 'Subclasses must define "model_name" class variable'
            raise TypeError(msg)
        return super().__new__(cls, name, bases, attrs)


class EmbeddingModel(metaclass=EmbeddingModelMeta):
    """
    Our embedding model class needs to create any overall variables in its
    init method and process images in a infer method.
    """

    # required image size
    patch = 224
    # default magnification
    magnification = 20
    # define model name in the subclasses
    # model_name = 'some model name'

    @abc.abstractmethod
    def infer(self, imgs: list[np.ndarray]) -> np.ndarray:
        """
        Calculate a set of embeddings for a set of images.

        :param imgs: an python array of images, all numpy arrays of identical
            shape (patch, patch, at least 3).
        :returns: a numpy array of shape (number of images, length of embedding
            vector)
        """


class GigapathModel(EmbeddingModel):
    model_name = 'prov-gigapath/prov-gigapath'

    def __init__(self):
        import timm
        import torch
        from torchvision import transforms

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tile_encoder = timm.create_model(f'hf_hub:{self.model_name}', pretrained=True)
        self.tile_encoder = self.tile_encoder.to(self.device)
        self.tile_encoder.eval()

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def infer(self, imgs: list[np.ndarray]) -> np.ndarray:
        import torch

        imgs = [np.copy(img) if not img.flags['WRITEABLE'] else img for img in imgs]
        imgstensor = torch.stack([self.transformer(img[:, :, :3]) for img in imgs])
        imgstensor = imgstensor.to(self.device)
        return self.tile_encoder(imgstensor).to('cpu').numpy()


class UNIModel(EmbeddingModel):
    """
    MahmoodLab/UNI has specific license requirements; please make sure you
    honor them.  It cannot be used commercially without approval.
    """

    model_name = 'MahmoodLab/UNI'

    def __init__(self):
        import timm
        import torch
        from torchvision import transforms

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tile_encoder = timm.create_model(
            f'hf_hub:{self.model_name}', pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.tile_encoder = self.tile_encoder.to(self.device)
        self.tile_encoder.eval()

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        ])

    def infer(self, imgs: list[np.ndarray]) -> np.ndarray:
        import torch

        imgs = [np.copy(img) if not img.flags['WRITEABLE'] else img for img in imgs]
        imgstensor = torch.stack([self.transformer(img[:, :, :3]) for img in imgs])
        imgstensor = imgstensor.to(self.device)
        return self.tile_encoder(imgstensor).to('cpu').numpy()


class DinoV2LargeModel(EmbeddingModel):
    model_name = 'facebook/dinov2-large'

    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

    def infer(self, imgs: list[np.ndarray]) -> np.ndarray:
        import torch

        imgstensor = torch.stack([torch.from_numpy(img[:, :, :3]) for img in imgs])
        inputs = self.processor(images=imgstensor, return_tensors='pt').to(self.device)
        results = self.model(**inputs)
        results = torch.mean(results.last_hidden_state, dim=1)
        return results.to('cpu').numpy()


class MidnightModel(EmbeddingModel):
    model_name = 'kaiko-ai/midnight'

    def __init__(self):
        import torch
        from torchvision import transforms
        from transformers import AutoModel

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def infer(self, imgs: list[np.ndarray]) -> np.ndarray:
        import torch

        imgs = [np.copy(img) if not img.flags['WRITEABLE'] else img for img in imgs]
        imgstensor = torch.stack([self.transformer(img[:, :, :3]) for img in imgs])
        imgstensor = imgstensor.to(self.device)
        results = self.model(imgstensor).last_hidden_state
        cls_embedding, patch_embeddings = results[:, 0, :], results[:, 1:, :]
        results = torch.cat([cls_embedding, patch_embeddings.mean(1)], dim=-1)
        return results.to('cpu').numpy()
