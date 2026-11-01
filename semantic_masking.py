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

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Create threshold mask
_, mask = cv2.threshold(
    gray,
    35,
    255,
    cv2.THRESH_BINARY
)

# Apply mask
masked_image = cv2.bitwise_and(
    image,
    image,
    mask=mask
)

# Create a 3x3 kernel
kernel = np.ones((7, 7), np.uint8)

# Apply morphological closing
cleaned_mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(mask, cmap="gray")
axes[0].set_title("Original Mask")

axes[1].imshow(cleaned_mask, cmap="gray")
axes[1].set_title("Mask After Closing")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

# Cleaned image
cleaned_image = cv2.bitwise_and(
    image,
    image,
    mask=cleaned_mask
)

plt.imshow(cleaned_image)
plt.title("Final Masked Image")
plt.axis("off")
plt.show()

# Calcualte percentage of pixals masked
foreground_pixels = np.count_nonzero(mask)
total_pixels = mask.size
percentage = (foreground_pixels / total_pixels) * 100
print(f"Foreground: {percentage:.2f}%")
