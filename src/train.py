import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from utils import extract_landmarks

DATASET_DIR = "data"
MODEL_DIR = "model"

X = []
y = []

# Walk through dataset/<label>/<image>.jpg, extracting landmarks for each image
for label in os.listdir(DATASET_DIR):
    label_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    for image_name in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_name)
        landmarks = extract_landmarks(image_path)

        # Skip images where no hand was detected
        if landmarks is None:
            continue

        X.append(landmarks)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Encode string labels (e.g. "A", "B", ...) as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Build a simple feed-forward classifier
num_classes = len(label_encoder.classes_)
model = keras.Sequential(
    [
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

# Save the trained model and label encoder
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, "asl_model.keras"))
np.save(os.path.join(MODEL_DIR, "label_encoder.npy"), label_encoder.classes_)
