import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Step 1: Grayscale Conversion (if the image is not already in grayscale)
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Gaussian Blur
def gaussian_blur(image, kernel_size=5, sigma=1.4):
    # Create Gaussian kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    sum_val = 0.0
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = np.exp(-((i - kernel_size // 2) ** 2 + (j - kernel_size // 2) ** 2) / (2 * sigma ** 2))
            sum_val += kernel[i, j]
    
    kernel /= sum_val  # Normalize kernel
    return cv2.filter2D(image, -1, kernel)

# Step 3: Compute Gradients using Sobel Filters
def sobel_filters(image):
    # Sobel Kernels for gradient calculation
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Apply Sobel filters to get gradients in the x and y directions
    gradient_x = cv2.filter2D(image, -1, Gx)
    gradient_y = cv2.filter2D(image, -1, Gy)

    # Compute gradient magnitude and angle
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_angle = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    gradient_angle[gradient_angle < 0] += 180  # Normalize angle to [0, 180]
    
    return gradient_magnitude, gradient_angle

# Step 4: Non-Maximum Suppression
def non_maximum_suppression(gradient_magnitude, gradient_angle):
    height, width = gradient_magnitude.shape
    suppressed_image = np.zeros_like(gradient_magnitude, dtype=np.float32)

    # For each pixel, check its gradient direction and suppress non-maxima
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = gradient_angle[i, j]
            magnitude = gradient_magnitude[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbor1 = gradient_magnitude[i, j + 1]
                neighbor2 = gradient_magnitude[i, j - 1]
            elif (22.5 <= angle < 67.5):
                neighbor1 = gradient_magnitude[i + 1, j + 1]
                neighbor2 = gradient_magnitude[i - 1, j - 1]
            elif (67.5 <= angle < 112.5):
                neighbor1 = gradient_magnitude[i + 1, j]
                neighbor2 = gradient_magnitude[i - 1, j]
            else:  # (112.5 <= angle < 157.5)
                neighbor1 = gradient_magnitude[i - 1, j + 1]
                neighbor2 = gradient_magnitude[i + 1, j - 1]

            # Suppress non-maxima
            if magnitude >= neighbor1 and magnitude >= neighbor2:
                suppressed_image[i, j] = magnitude
            else:
                suppressed_image[i, j] = 0

    return suppressed_image

# Step 5: Edge Tracking by Hysteresis
def edge_tracking_by_hysteresis(image, low_threshold, high_threshold):
    # Step 5a: Mark strong edges as 255 and weak edges as 75
    strong_edges = (image > high_threshold)
    weak_edges = ((image >= low_threshold) & (image <= high_threshold))

    # Step 5b: Iterate and connect weak edges that are connected to strong edges
    result_image = np.zeros_like(image, dtype=np.uint8)
    result_image[strong_edges] = 255

    # Use a 3x3 kernel to check the weak edges
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if weak_edges[i, j]:
                # Check if any of the 8 neighbors is a strong edge
                if np.any(result_image[i-1:i+2, j-1:j+2] == 255):
                    result_image[i, j] = 255

    return result_image

# Canny Edge Detection Implementation
def canny_edge_detection(image_path, low_threshold=50, high_threshold=150):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    gray_image = to_grayscale(image)
    
    # Step 1: Apply Gaussian Blur
    blurred_image = gaussian_blur(gray_image)
    
    # Step 2: Compute gradients
    gradient_magnitude, gradient_angle = sobel_filters(blurred_image)
    
    # Step 3: Apply Non-Maximum Suppression
    suppressed_image = non_maximum_suppression(gradient_magnitude, gradient_angle)
    
    # Step 4: Edge Tracking by Hysteresis
    final_edges = edge_tracking_by_hysteresis(suppressed_image, low_threshold, high_threshold)
    
    return final_edges

# Display the result
image_path = 'pic1.png'  # Replace with your image file
edges = canny_edge_detection(image_path)

# Show results using matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection (from Scratch)")
plt.axis('off')
plt.show()
