# We could use
#  FROM pytorch/pytorch:latest
# instead of using a python base.  This would let us skip setting the
# nvidia_driver label and avoid having to pip install a specific torch and cuda
# library version.  But -- the pytorch image is several gigabytes larger
# because it contains elements we don't use, and uses Python 3.10.  We can use
# a newer python, and still ask for the same version of cuda support that the
# pytorch/pytorch base would use.
FROM python:3.11-slim

# This tells girder_worker to enable gpu if possible.  Although not an official
# standard, it is the closest thing to standard way to indicate a docker image
# should be run with GPU, if possible.
LABEL com.nvidia.volumes.needed=nvidia_driver

# torch recommends installing torch, torchvision, and torchaudio, but skipping
# torchaudio does no harm.  The find ... line removes cached python files to
# keep the docker image smaller
RUN python -m pip --no-cache-dir install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    find / -xdev -name __pycache__ -type d -exec rm -r {} \+

LABEL maintainer="Kitware, Inc. <kitware@kitware.com>"

# These are all included in the dependencies, but by adding it here add a layer
# to the docker build process, making rebuilds of just our source faster
RUN python -m pip --no-cache-dir install large_image[sources] timm transformers --find-links https://girder.github.io/large_image_wheels && \
    find / -xdev -name __pycache__ -type d -exec rm -r {} \+

# Copy our source files to the docker and install them
COPY . /opt/main
WORKDIR /opt/main
RUN python -m pip --no-cache-dir install . --find-links https://girder.github.io/large_image_wheels && \
    find / -xdev -name __pycache__ -type d -exec rm -r {} \+

WORKDIR /opt/main/histomicstk_similarity

# These are just basic tests to prove that the build has worked to the very
# smallest degree
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli && \
    find / -xdev -name __pycache__ -type d -exec rm -r {} \+
RUN python -m slicer_cli_web.cli_list_entrypoint EmbeddingSimilarity --help && \
    find / -xdev -name __pycache__ -type d -exec rm -r {} \+

# This makes the results show up in a more timely manner
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
