import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands


def extract_landmarks(image_path):
    """
    Load an image, run it through MediaPipe Hands, and extract hand landmarks.

    Args:
        image_path: Path to the image file.

    Returns:
        A flat list of 63 floats (21 landmarks x, y, z) if a hand is
        detected, otherwise None.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    landmarks = results.multi_hand_landmarks[0]
    return [coord for lm in landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
