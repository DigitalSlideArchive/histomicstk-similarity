HistomicsTK Similarity
======================

Identify similar tissue within an image using foundational models.

This demonstrates using foundational models to determine similarity within a single image.  It is easily extensible to use a variety of models.

Development
-----------

This can be built like so::

    docker build --force-rm -t dsarchive/histomicstk_similarity .

And then installed in a Digital Slide Archive instance by navigating to Collections -> Tasks -> Slicer CLI Web Tasks -> Upload CLI button and importing "dsarchive/histomicstk_similarity:latest".
