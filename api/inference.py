import torch
import cv2
import numpy as np
from pytorchvideo.models.hub import x3d_m
from torchvision.transforms import Compose
from pytorchvideo.transforms import Normalize
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 9
FRAMES = 16
SIZE = 224

CLASS_NAMES = [
    "sneeze/cough",
    "staggering",
    "falling",
    "touch head (headache)",
    "touch chest (stomachache/heart pain)",
    "touch back (backache)",
    "touch neck (neckache)",
    "nausea/vomiting",
    "use a fan / feeling warm"
]

# Load model
print(f"🤖 Loading model on {DEVICE}...")
model = x3d_m(pretrained=False)
model.blocks[5].proj = torch.nn.Linear(model.blocks[5].proj.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("best_action_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully!")

# Simplified transform (no temporal subsampling since we control frames)
transform = Compose([
    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
])

def preprocess_video(frames):
    """
    Preprocess exactly 16 frames for inference
    IMPORTANT: No temporal subsampling here since we already have 16 frames
    """
    if len(frames) != FRAMES:
        raise ValueError(f"Expected {FRAMES} frames, got {len(frames)}")
    
    # Convert & Resize
    frames_processed = []
    for frame in frames:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to model input size
        frame_resized = cv2.resize(frame_rgb, (SIZE, SIZE))
        frames_processed.append(frame_resized)
    
    # Stack and convert to tensor
    clip = np.stack(frames_processed)  # (T, H, W, C)
    clip = torch.from_numpy(clip).float().permute(3, 0, 1, 2)  # (C, T, H, W)
    
    # Apply normalization
    clip = transform(clip)
    
    return clip.unsqueeze(0).to(DEVICE)  # Add batch dimension

def predict_action(frames, return_confidence=True):
    """
    Predict action from frames with optional confidence score
    
    Args:
        frames: List of 16 frames (numpy arrays)
        return_confidence: If True, returns (action, confidence)
    
    Returns:
        If return_confidence=True: (action_name, confidence_score)
        If return_confidence=False: action_name
    """
    clip = preprocess_video(frames)
    
    with torch.no_grad():
        logits = model(clip)
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        confidence = torch.max(probs).item()
        pred_idx = torch.argmax(logits, dim=1).item()
    
    action = CLASS_NAMES[pred_idx]
    
    if return_confidence:
        return action, confidence
    return action

def predict_action_with_probabilities(frames):
    """
    Get full probability distribution for all classes
    Useful for debugging and advanced smoothing
    """
    clip = preprocess_video(frames)
    
    with torch.no_grad():
        logits = model(clip)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}