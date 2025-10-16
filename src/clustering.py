from sklearn.cluster import Birch, AgglomerativeClustering


# We implement a hybrid clustering method that first uses Birch to reduce the number of points
# by clustering them into many small clusters (CF nodes), and then uses Agglomerative Clustering.
# The complexity remains linear in the number of samples, despite the quadratic complexity of Agglomerative Clustering
# in the number of mini-clusters.



def clustering(feature_vectors, threshold=30.0, n_clusters=None, use_hybrid=True):
    if not use_hybrid:
        # for small datasets, we can directly use Agglomerative Clustering
        agg_model = AgglomerativeClustering(
            n_clusters=n_clusters, 
            distance_threshold=threshold,
            linkage='ward' 
        )
        return agg_model.fit_predict(feature_vectors)
        
    # for big datasets, we use the hybrid method
    
    # Use Birch to rapidly find initial tight clusters (CF nodes)
    # The threshold here should be small to create many micro-clusters (for the provided dataset, 1.5 works well),
    # but for larger datasets, bigger mini-clusters are needed to keep the number of micro-clusters relatively low.
    BIRCH_THRESHOLD = 1.5
    
    birch_preprocessor = Birch(
        threshold=BIRCH_THRESHOLD, 
        n_clusters=None, 
        branching_factor=50
    )
    birch_preprocessor.fit(feature_vectors)

    # Get the features of the micro-cluster centers found by Birch
    cluster_centers = birch_preprocessor.subcluster_centers_
    
    # Use Agglomerative Clustering on the smaller set of centers

    agg_model = AgglomerativeClustering(
        n_clusters=n_clusters, 
        distance_threshold=threshold,
        linkage='ward' 
    )

    agg_model.fit(cluster_centers)
    
    # Get the final label for each Birch micro-center
    final_center_labels = agg_model.labels_

    # Use the full labels array from the fitted Birch model to map
    # the original data points (using the .labels_ attribute) to the final label.
    # Note: birch_preprocessor.labels_ has length N, giving the index of the CF node for each point.
    original_data_labels = final_center_labels[birch_preprocessor.labels_] # Use .labels_

    return original_data_labels



