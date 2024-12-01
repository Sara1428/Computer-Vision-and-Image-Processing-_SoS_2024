import numpy as np
import cv2
import matplotlib.pyplot as plt

def initialize_centroids(pixels, k):
    np.random.seed(0)
    indices = np.random.choice(pixels.shape[0], k, replace=False)
    centroids = pixels[indices]
    return centroids

def compute_distances(pixels, centroids):
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
    return distances

def assign_pixels_to_centroids(distances):
    return np.argmin(distances, axis=1)

def recompute_centroids(pixels, labels, k):
    new_centroids = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans_segmentation(image, k=2, max_iters=100, tol=1e-4):
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    centroids = initialize_centroids(pixels, k)
    
    for i in range(max_iters):
        distances = compute_distances(pixels, centroids)
        labels = assign_pixels_to_centroids(distances)
        new_centroids = recompute_centroids(pixels, labels, k)

        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        
        centroids = new_centroids

    segmented_pixels = centroids[labels]
    segmented_image = segmented_pixels.reshape(image.shape)
    
    return segmented_image

image = cv2.imread('pic.png')  
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

segmented_image = kmeans_segmentation(image_rgb, k=3)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title(f"Segmented Image (k=3)")
plt.axis('off')
plt.tight_layout()
plt.show()
