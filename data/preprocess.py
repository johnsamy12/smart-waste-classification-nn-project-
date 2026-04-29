import os
import cv2
import numpy as np

DATA_DIR = "."
IMG_SIZE = 224

X = []
y = []

labels = [f for f in os.listdir(DATA_DIR) if os.path.isdir(f)]

for i, label in enumerate(labels):
    folder = os.path.join(DATA_DIR, label)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = cv2.imread(path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X.append(img)
        y.append(i)

X = np.array(X)
y = np.array(y)

print("Images:", X.shape)
print("Labels:", y.shape)