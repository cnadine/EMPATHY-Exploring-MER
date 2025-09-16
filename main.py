from transformers import pipeline
from deepface import DeepFace
import cv2
import numpy as np

# === Text Emotion Pipeline ===
print("Loading text emotion model...")
text_pipe = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None  # Return all emotion scores
)

# === Facial Emotion Setup ===
# DeepFace supports: 'emotion', 'age', 'gender', 'race'
# Emotions: 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
print("Initializing webcam for facial emotion (press SPACE to capture, ESC to skip)...")

# Map DeepFace emotions to text model emotions for fusion
EMOTION_MAPPING = {
    'angry': 'anger',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'joy',
    'sad': 'sadness',
    'surprise': 'surprise',
    'neutral': 'neutral'
}

# Text model's emotion labels (for alignment)
TEXT_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

def capture_face_emotion():
    """Capture image from webcam and return emotion scores."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("📸 Press SPACE to capture face, ESC to skip.")
    face_emotions = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Capture Face for Emotion (Press SPACE or ESC)', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            print("⏭️  Skipping facial input.")
            break
        elif key == 32:  # SPACE key
            try:
                # Analyze emotion from captured frame
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list):
                    result = result[0]  # Take first face if multiple detected

                # Normalize DeepFace emotion scores
                face_emotions_raw = result['emotion']
                total = sum(face_emotions_raw.values())
                face_emotions_normalized = {k: v / total for k, v in face_emotions_raw.items()}

                # Map to text model labels
                face_emotions = {EMOTION_MAPPING.get(k, k): v for k, v in face_emotions_normalized.items()}

                # Ensure all text emotions are present (default 0)
                face_emotions = {e: face_emotions.get(e, 0.0) for e in TEXT_EMOTIONS}

                print("✅ Facial emotion captured!")
                break

            except Exception as e:
                print(f"⚠️  Face analysis failed: {e}")
                face_emotions = None
                break

    cap.release()
    cv2.destroyAllWindows()
    return face_emotions

def fuse_emotions(text_scores, face_scores, text_weight=0.5, face_weight=0.5):
    """Fuse text and face emotion scores using weighted average."""
    if face_scores is None:
        return text_scores  # Fallback to text only

    fused = {}
    for emotion in TEXT_EMOTIONS:
        t_score = text_scores.get(emotion, 0.0)
        f_score = face_scores.get(emotion, 0.0)
        fused[emotion] = text_weight * t_score + face_weight * f_score

    # Normalize fused scores to sum to 1.0
    total = sum(fused.values())
    if total > 0:
        fused = {k: v / total for k, v in fused.items()}
    return fused

# === Main Loop ===
print("\n" + "="*60)
print("🧠 MULTIMODAL EMOTION RECOGNITION SYSTEM")
print("   Modalities: Text + Facial Expression")
print("="*60)
print("Type a sentence and press Enter. Type 'quit' to exit.\n")

while True:
    text = input("You: ")
    if text.lower() == "quit":
        print("👋 Exiting...")
        break

    # Get text emotion scores
    text_results = text_pipe(text)[0]
    text_scores = {r['label']: r['score'] for r in text_results}

    # Capture facial emotion (optional)
    face_scores = capture_face_emotion()

    # Fuse modalities
    fused_scores = fuse_emotions(text_scores, face_scores)

    # Sort by confidence
    sorted_emotions = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "-"*40)
    print("📊 FUSED EMOTION PREDICTIONS:")
    for emotion, score in sorted_emotions:
        print(f"  {emotion:15} {score:.3f}")

    if face_scores is not None:
        print(f"\n📌 Fusion weights: Text({0.5}) + Face({0.5})")
    else:
        print("\n📌 Only text modality used.")

    print()