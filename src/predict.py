import os
import tempfile

import cv2
import numpy as np
from tensorflow import keras

from utils import extract_landmarks

MODEL_PATH = "model/asl_model.keras"
LABEL_ENCODER_PATH = "model/label_encoder.npy"

model = keras.models.load_model(MODEL_PATH)
classes = np.load(LABEL_ENCODER_PATH, allow_pickle=True)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()
    cv2.imwrite(tmp_path, frame)
    landmarks = extract_landmarks(tmp_path)
    os.remove(tmp_path)

    if landmarks is not None:
        predictions = model.predict(np.array([landmarks]), verbose=0)[0]
        top_index = np.argmax(predictions)
        label = classes[top_index]
        confidence = predictions[top_index] * 100

        cv2.putText(
            frame,
            f"{label} ({confidence:.1f}%)",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("SignLink", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
