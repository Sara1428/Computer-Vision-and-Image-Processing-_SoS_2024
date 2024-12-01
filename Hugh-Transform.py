import numpy as np
import cv2
import matplotlib.pyplot as plt

def edge_detection(image):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = cv2.filter2D(image, -1, sobel_x)
    gradient_y = cv2.filter2D(image, -1, sobel_y)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    return magnitude

def hough_transform(image, threshold=100):
    edges = edge_detection(image)
    diag_len = int(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
    max_r = diag_len
    max_theta = np.pi
    rho_res = 1
    theta_res = np.pi / 180
    accumulator = np.zeros((2 * max_r, int(max_theta / theta_res)), dtype=np.int)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if edges[y, x] > 0:
                for theta_idx in range(accumulator.shape[1]):
                    theta = theta_idx * theta_res
                    rho = int(x * np.cos(theta) + y * np.sin(theta))
                    rho_idx = int(rho + max_r)
                    accumulator[rho_idx, theta_idx] += 1

    lines = []
    for rho_idx in range(accumulator.shape[0]):
        for theta_idx in range(accumulator.shape[1]):
            if accumulator[rho_idx, theta_idx] >= threshold:
                rho = rho_idx - max_r
                theta = theta_idx * theta_res
                lines.append((rho, theta))

    return lines, edges

def draw_detected_lines(image, lines):
    image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image_copy

image = cv2.imread('pic1.png', cv2.IMREAD_GRAYSCALE)
lines, edges = hough_transform(image, threshold=100)
image_with_lines = draw_detected_lines(image, lines)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection")
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(image_with_lines)
plt.title("Detected Lines (Hough Transform)")
plt.axis('off')
plt.show()
