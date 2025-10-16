We do a hybrid clustering algorithm to group logos by similarity. The algorithm is a
combination of BIRCH with agglomerative clustering, for complexity reasons (we do a
mini-clustering with BIRCH, which is linear, and then we do agglomerative clustering
on these mini-clusters, which is quadratic in the number of mini-clusters, but we can
controll this number). After preprocessing each image (most importantly rescaling it
to 128x128), we embed it in a high dimensional euclidean space (1031 dimensions), where
the first 512 entries represent the linearized HOG (histogram of oriented gradients,
for edge recognintion) of the logo, the next 7 entries are the HU moments, and the rest
of the 512 entries are a HSV histogram for color representation.

To run this project:

1. Run setup.py: this will download the logos listed in data/logos.snappy.parquet and
   store them in data/logos. There should be around 3000 logos downloaded.
2. Run main.py: here will happen the actual clustering. A file named 'clustering_results.csv'
   will be created inside data/, containing a dataframe pairing each logo path to a cluster
   number.
3. For testing the results, run testing.py
