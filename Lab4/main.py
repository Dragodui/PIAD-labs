import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.cluster import KMeans
import cv2


# 1. Discretion
def generate_discrete_signal(f, fs):
    t = np.arange(0, 1, 1 / fs)
    s = np.sin(2 * np.pi * f * t)
    return t, s


def discretion():
    f = 100
    sampling_frequencies = [20, 21, 45, 50, 150, 200, 250, 1000]

    for fs in sampling_frequencies:
        t, s = generate_discrete_signal(f, fs)
        plt.plot(t, s, label=f"f_s = {fs} Hz")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Discrete-Time Sinusoid")
    plt.legend()
    plt.grid(True)
    plt.show()


# 2. Quantization

def quantize_image(image, num_colors):
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    flat_image = image.reshape((-1, 1))

    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(flat_image)
    cluster_centers = kmeans.cluster_centers_

    quantized_image = cluster_centers[kmeans.predict(flat_image)].reshape(image.shape)

    return quantized_image


def quantization():
    image = io.imread("imgs/image.jpg")

    num_colors_list = [2, 4, 8, 16]
    for num_colors in num_colors_list:
        quantized_image = quantize_image(image.copy(), num_colors)

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(image, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(quantized_image, cmap="gray")
        plt.title(f"Quantized Image ({num_colors} Colors)")
        plt.axis("off")

    plt.show()


# 3. Binarization

def binarize_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

    return img, binary


def binarization():
    image_path = 'imgs/image.jpg'
    original_image, binary_image = binarize_image(image_path)

    cv2.imshow('Original Image', original_image)
    cv2.imshow('Binary Image', binary_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
