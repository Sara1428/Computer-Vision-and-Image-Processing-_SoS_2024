import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('pic1.png', cv2.IMREAD_GRAYSCALE)  

if image is None:
    print("Error: Image not found")
    exit()

blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)


plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')

plt.tight_layout()
plt.show()
