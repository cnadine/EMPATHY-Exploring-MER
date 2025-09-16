# Multimodal Emotion Recognition

Combines **text** and **facial expression** to predict emotion using:

- Hugging Face Transformers (text)
- DeepFace + OpenCV (facial expression)

## Setup

```bash
pip install transformers torch opencv-python deepface numpy
python main.py
```

## How to Use
1. Type a sentence and press ENTER.
2. Press SPACE to capture face from webcam (or ESC to skip).
3. See fused emotion scores!
