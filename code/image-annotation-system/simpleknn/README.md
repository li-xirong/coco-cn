simpleknn
=========

Find k nearest neighbors by an exhaustive search, used for content-based image retrieval


Given an image collection say ``toydata`` with ``n`` image, we presume that a specific visual feature, named as ``f1``, has been extracted and stored as ``toydata/FeatureData/f1/id.feature.txt``, where each line starts with a unique image id followed by its feature vector. Given a test image and its feature vector, simpleknn finds the ``k`` nearest neighbors from ``toydata`` by computing a given distance, namely ``l1`` or ``l2``, between the feature vectors.

See ``test.sh`` for usage.
