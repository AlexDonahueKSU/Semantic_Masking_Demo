import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Locate our project directory
BASE_DIR = Path(__file__).resolve().parent

# Locate our sample image
IMAGE_PATH = BASE_DIR / "images" / "original" / "nut_good.png"

# Load image
image = cv2.imread(str(IMAGE_PATH))

# Check if image loaded successfully
if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# Convert BGR to RGB for displaying
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display image
plt.imshow(image)
plt.title("Original Metal Nut")
plt.axis("off")
plt.show()

# Inspect image properties
print("Image dimensions:", image.shape)
print("Data type:", image.dtype)
print("Pixel values:", image[0, 0])

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Display grayscale image
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

# Generate binary mask using thresholding
_, mask = cv2.threshold(
    gray,
    50,
    255,
    cv2.THRESH_BINARY
)

# Display mask
plt.imshow(mask, cmap="gray")
plt.title("Binary Mask")
plt.axis("off")
plt.show()

# Apply mask to original image
masked_image = cv2.bitwise_and(
    image,
    image,
    mask=mask
)

# Display result
plt.imshow(masked_image)
plt.title("Masked Image")
plt.axis("off")
plt.show()
