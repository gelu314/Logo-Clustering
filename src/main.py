import pandas as pd
import numpy as np
import os
import glob
import cv2 as cv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from embedding import embed_image
from clustering import clustering


img_size = (128, 128)



def main(threshold = .5, n_clusters = None):

    # we import the logo paths into a list
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_dir = os.path.join(current_dir, '..', 'data', 'logos')

    png_pattern = os.path.join(logo_dir, '*.png')
    jpg_pattern = os.path.join(logo_dir, '*.jpg')

    all_logo_files = glob.glob(png_pattern) + glob.glob(jpg_pattern)
    if not all_logo_files:
        print("No logo files found in the specified directory.")
        return

    n_logos = len(all_logo_files)
    print(f"Found {n_logos} logo files")

    feature_vectors = []
    processed_files = []

    # now we do the embedding for each logo in a high dimensional euclidean space
    # and store the feature vectors in feature_vectors

    for i, file_path in enumerate(all_logo_files):
        try:
            image = cv.imread(file_path, cv.IMREAD_COLOR)
            if image is None:
                print(f"Warning: Unable to load image at {file_path}. Skipping.")
                continue

            if len(image.shape) < 3:
                image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
                print(f"Converted grayscale image to BGR for {file_path}")

            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            feature_vector, hog_len, hu_len, hsv_len = embed_image(image_rgb)
            
            feature_vectors.append(feature_vector)
            processed_files.append(file_path)

            print(f"Processed {i+1}/{len(all_logo_files)}: {file_path} -> Feature vector shape: {feature_vector.shape}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


    feature_vectors = np.array(feature_vectors)
    print(f"Total processed images: {len(processed_files)}")

    # the 7 hu moments entries have values in range [-10, 10], and the ~1024 rest of entries are in range [0, 1]
    # so we normalize all vectors to have 0 mean and unit variance

    scaler = StandardScaler()
    feature_vectors = scaler.fit_transform(feature_vectors)

    # we need to decide how important each type of feature is
    # intuitively, all three types of features are important
    # but shapes and structures yield way stronger resemblence to the human eye than colors
    # so we give more importance to hog and hu moments features

    hog_imp = .4
    hu_imp = .4
    hsv_imp = .2

    # now we compute weights for each type of feature (the hog weight is kept at 1 as refference for the others)
    
    hog_weight = 1
    hu_weight = np.sqrt((hog_len/hu_len) * (hu_imp/hog_imp))
    hsv_weight = np.sqrt( (hog_len/hsv_len) * (hsv_imp/hog_imp))

    feature_vectors[:, :hog_len] *= hog_weight
    feature_vectors[:, hog_len:hog_len+hu_len] *= hu_weight
    feature_vectors[:, hog_len+hu_len:] *= hsv_weight

    #reducing dimensions (1031 is too high):
    pca = PCA(n_components=100)
    feature_vectors = pca.fit_transform(feature_vectors)

    # some checks:
    # print(f"Feature vectors shape after dimension reduction: {feature_vectors.shape}") 
    # print(f"Feature vector sample (first 10 values): {feature_vectors[0][:10]}")
    # print(f"Feature vector min/max (first vector): {feature_vectors[0].min()}/{feature_vectors[0].max()}")   
    # print(f"hog weight was {hog_weight}, hu weight was {hu_weight} and hsv weight was {hsv_weight}")

    # now we cluster the feature vectors
    labels = clustering(feature_vectors, threshold=threshold, n_clusters=None)
    print(f"clustering was done with threshold {threshold}")

    #saving the results to clustering_results.csv
    results_df = pd.DataFrame({
        'file_path': processed_files,
        'label': labels
    })

    data_dir = os.path.join(current_dir, '..', 'data')
    output_path = os.path.join(data_dir, 'clustering_results.csv')

    try:
        results_df.to_csv(output_path, index=False)
        print(f"Clustering results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")

    print(f"First 100 Labels: {labels[:100]}")
    print(len(set(labels)), " unique clusters found")



if __name__ == "__main__":
    main(33.0)  # you can adjust the threshold or set n_clusters as needed
    # the threshold is the minimum distance between two clusters for them to be merged,
    # so higher threshold means less clusters
    # for the given dataset, 33 works well, but for bigger datasets, higher thresholds are needed



