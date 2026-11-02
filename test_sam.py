import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

# Select GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SAM
sam = sam_model_registry["vit_b"](
    checkpoint="models/sam_vit_b_01ec64.pth"
)

sam.to(device=device)

# Create predictor
predictor = SamPredictor(sam)

# Load image
image = cv2.imread("images/original/nut_good.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Give image to SAM
predictor.set_image(image)

# --- Generate bounding box using OpenCV ---

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

_, mask = cv2.threshold(
    gray, 35, 255, cv2.THRESH_BINARY
)

kernel = np.ones((7, 7), np.uint8)

cleaned_mask = cv2.morphologyEx(
    mask, cv2.MORPH_CLOSE, kernel
)

contours, hierarchy = cv2.findContours(
    cleaned_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

largest_contour = max(contours, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(largest_contour)

# Bounding box around the nut
input_box = np.array([
    [x, y, x + w, y + h]
])

# Generate masks
masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=True
)

for i, score in enumerate(scores):
    print(f"Mask {i + 1} predicted IoU: {score:.4f}")

# Find the index of the highest-scoring mask
best_index = np.argmax(scores)

# Select the best mask
best_mask = masks[best_index]

print("Selected mask:", best_index + 1)
print("Selected score:", scores[best_index])

# Apply selected mask
sam_masked_image = cv2.bitwise_and(
    image,
    image,
    mask=best_mask.astype(np.uint8) * 255
)

# Display result
plt.imshow(sam_masked_image)
plt.title("SAM Selected Mask")
plt.axis("off")
plt.show()

# Display the three possible mask
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

axes[0].imshow(image)
axes[0].scatter(350, 350, color="red", marker="*", s=150)
axes[0].set_title("Original Image")

for i in range(3):
    axes[i + 1].imshow(image)
    axes[i + 1].imshow(masks[i], alpha=0.5, cmap="jet")
    axes[i + 1].set_title(f"Mask {i + 1} (Score: {scores[i]:.3f})")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
