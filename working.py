
import matplotlib.pyplot as plt
import cv2

import pandas as pd
import numpy as np

df=pd.read_csv('boxes.csv')

# Load image
image = cv2.imread("input.jpg")

# Choose a model type + checkpoint
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# Set image
predictor.set_image(image)

# Define a bounding box prompt: (x_min, y_min, x_max, y_max)
box = [100, 50, 400, 350]  # example values
masks, scores, logits = predictor.predict(box=box)

# Save mask
mask = masks[0] * 255
cv2.imwrite("mask.png", mask.astype("uint8"))