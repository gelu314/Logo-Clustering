from skimage.feature import hog
from skimage import data
import cv2 as cv
import numpy as np
'''
the embedding function takes in an image and returns a high dimensional feature vector
consisting of HOG features, Hu Moments, and color histogram features.
'''
def embed_image(image):

    # here we check if image is rgb or grayscale
    if len(image.shape) == 3:
        image_for_hsv = image 
        gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    elif len(image.shape) == 2:
        gray_image = image 
        image_for_hsv = cv.cvtColor(gray_image, cv.COLOR_GRAY2RGB)
    else:
        raise ValueError("Input image must be 2D (grayscale) or 3D (color).")


    fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    hog_vector = fd.flatten()

    moments = cv.moments(gray_image)
    hu_vector = cv.HuMoments(moments).flatten()
    hu_vector = -np.sign(hu_vector) * np.log10(np.abs(hu_vector) + 1e-10)

    hsv_image = cv.cvtColor(image_for_hsv, cv.COLOR_RGB2HSV) 
    hsv_vector = cv.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
    hsv_vector = cv.normalize(hsv_vector, hsv_vector, norm_type=cv.NORM_L1)

    feature_vector = np.concatenate([hog_vector, hu_vector, hsv_vector])
    return feature_vector, len(hog_vector), len(hu_vector), len(hsv_vector)

if __name__ == "__main__":
    image=data.checkerboard()
    feature_vector = embed_image(image)
    print("Feature vector shape:", feature_vector.shape)

