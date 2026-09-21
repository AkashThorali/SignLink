## Overview
Real-time American Sign Language (ASL) alphabet recognition using MediaPipe hand 
landmarks and a TensorFlow classifier. The model extracts 21 hand landmarks 
(63 features) per frame and classifies them across 29 ASL signs (A–Z, space, 
delete, nothing).

## Demo

![SignLink Demo](assets/demo.gif)

## Architecture
- Hand detection: MediaPipe Hands (static_image_mode for training, live feed for inference)
- Feature extraction: 21 landmarks × (x, y, z) = 63 features per hand
- Classifier: 3-layer dense neural network (Dense 64 → Dense 64 → Dense 29, softmax output)
- Training data: ASL Alphabet Dataset (87,000 images, 29 classes)

## Results
- Validation accuracy: 98.8% across 29 classes
- Training time: ~50 epochs on CPU

## Setup
1. Clone the repo
2. Create a virtual environment: python3 -m venv venv && source venv/bin/activate
3. Install dependencies: pip install -r requirements.txt
4. Download the ASL Alphabet dataset from Kaggle and place it at:
   data/archive/asl_alphabet_train/asl_alphabet_train/<label>/
5. Train the model: PYTHONPATH=src python3 src/train.py
6. Run live inference: PYTHONPATH=src python3 src/predict.py

## Known Limitations
- Each webcam frame is written to a temp file before landmark extraction; 
  refactoring extract_landmarks() to accept numpy arrays directly would reduce latency
- Trained on clean studio images; performance may vary in different lighting conditions

## Dataset
ASL Alphabet Dataset by Akash (Kaggle)
