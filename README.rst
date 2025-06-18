HistomicsTK Similarity
======================

Identify similar tissue within an image using foundational models.

This demonstrates using foundational models to determine similarity within a single image.  It is easily extensible to use a variety of models.

Basic Usage
-----------

This assumes you have an instance of Digital Slide Archive and have installed the HistomicsTK Similarity task and specified how to access the model you want to use.

- Open an image in the HistomicsUI

- Select Analyses -> dsarchive/histomics_similarity -> latest -> EmbeddingSimilarity.  You'll see the task parameters on the left side of your window.

- Pick a model in the Generation section.  The defaults will work, but if you use a large image or have a slow GPU, the initial data generation can take a while.

- Click "Submit".  Feature embeddings will be calculated for your image and stored in your Private folder under the name "Embedding Similarity-embedout.npz".  This can take a while depending on image size, model, and your worker computer's specifications.  A heatmap annotation will appear.

- To update an existing heatmap annotation, select "Embedding Input File" and select the file that was generated in a previous step.

- Make sure the annotation you want to update is visible.  From the annotation settings dialog (the gear icon next to the annotation name), copy the annotation Unique ID and paste it in the Annotation ID field.

- Click the dot icon in the Key Point control, then click on your image.

- Click "Submit".  The heatmap will update after a few seconds.

Installation
------------

Install in a Digital Slide Archive instance by navigating to Collections -> Tasks -> Slicer CLI Web Tasks -> Upload CLI button and importing "dsarchive/histomicstk_similarity:latest".

The task will appear in Tasks -> Slicer CLI Web Tasks -> dsarchive/histomicstk_similarity -> latest -> EmbeddingSimilarity and in the HistomicsUI Analyses menu.

Model Access
------------

By default the task will download foundational models from huggingface.co if they are not available.  Some of the models are gated or license-restriced and required authorization, which is based on your huggingface.co login and token.  These models could be downloaded separately and cached and used from the cache, or can be download as needed.  Either you need to provide credentials to the task or provide the cache directory to the task.  This can be configured in your ``docker-compose.override.yaml`` used to start your Digital Slide Archive instance.

For example, to use a cache that has been preloaded, you can add to your ``docker-compose.override.yaml`` file::

    ---
    services:
      worker:
        environment:
          GIRDER_WORKER_DOCKER_RUN_OPTIONS: '{"volumes": ["/some/path/on/the/worker/hfcache:/root/.cache"], "environment": {"HF_HUB_OFFLINE": "1"}}'

Instead, you could provide your ``HF_TOKEN``, but since this exposes it to all tasks, it is more secure to prepopulate the cache (see huggingface.co documentation on how to do so).  Besides doing it manually, this could also be done by running ``docker run --rm -it --env HF_TOKEN="$HF_TOKEN" -v /some/path/on/the/worker/hfcache:/root/.cache dsarchive/histomicstk_similarity:latest EmbeddingSimilarity /opt/main/tests/sample_Easy1.jpeg`` assuming you have your ``HF_TOKEN`` in your environment and adjust the cache path appropriately.

Development
-----------

This can be built like so::

    docker build --force-rm -t dsarchive/histomicstk_similarity .

And then installed in a Digital Slide Archive instance by navigating to Collections -> Tasks -> Slicer CLI Web Tasks -> Upload CLI button and importing "dsarchive/histomicstk_similarity:latest".
