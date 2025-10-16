import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import os

#we test the results from clustering_resutls.csv by visualizing some logos from a random cluster

def test_results():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_dir = os.path.join(current_dir, '..', 'data', 'clustering_results.csv')
    df = pd.read_csv(logo_dir)
    print(f"Total logos in results: {len(df)}")
    
    unique_labels = df['label'].unique()
    print(f"Number of clusters: {unique_labels.shape[0]}")
    
    random_cluster_id = np.random.choice(unique_labels)
    print(f"random_cluster_id: {random_cluster_id}")
    
    cluster_df = df[df['label'] == random_cluster_id]
    cluster_df = cluster_df.sample(frac=1).reset_index(drop=True)
    print(f"Number of logos in cluster {random_cluster_id}: {len(cluster_df)}")

    n_to_display = min(28, len(cluster_df))
    plt.figure(figsize=(12, 7))
    plt.title(f"Cluster {random_cluster_id}")
    for i in range(n_to_display):
        file_path = cluster_df['file_path'].iloc[i]
        image = cv.imread(file_path, cv.IMREAD_COLOR)
        if image is None:
            print(f"Warning: Unable to load image at {file_path}. Skipping.")
            continue
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        plt.subplot(7, 4, i+1)
        plt.imshow(image_rgb)
        plt.axis('off')
    plt.show()  

if __name__ == "__main__":
    test_results()

