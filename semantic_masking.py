import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load an image
image = cv2.imread("path/to/your/image.png")

# Check whether the image loaded successfully
if image is None:
    print("Could not load image!")
else:
    print("Image loaded successfully!")

    # Display image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
