import cv2
import numpy as np

# Load the image
image = cv2.imread("C:/Users/SAI VISHWA/Desktop/word-segment-final/word_segmentation-master/test.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform adaptive thresholding
thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)

# Sharpen the image (you can adjust the kernel size for stronger or weaker sharpening)
sharpened = cv2.GaussianBlur(thresholded, (0, 0), 3)
sharpened = cv2.addWeighted(thresholded, 1.5, sharpened, -0.5, 0)

# Adjust brightness (you can change the value to adjust brightness)
brightened = cv2.convertScaleAbs(sharpened, alpha=2, beta=10)

# Noise removal using Gaussian blur
denoised = cv2.GaussianBlur(brightened, (3, 3), 0)

# Perform morphological operations for noise removal with a larger kernel
kernel = np.ones((5, 5), np.uint8)
cleaned = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

# Save the preprocessed image
cv2.imwrite('preprocessed_image.png', cleaned)

# Display the preprocessed image
# cv2.imshow('Preprocessed Image', cleaned)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
