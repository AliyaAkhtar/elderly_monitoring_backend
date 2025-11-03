import torch
import cv2
import numpy as np
from pytorchvideo.models.hub import x3d_m
from torchvision.transforms import Compose
from pytorchvideo.transforms import UniformTemporalSubsample, Normalize

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
model = x3d_m(pretrained=False)
model.blocks[5].proj = torch.nn.Linear(model.blocks[5].proj.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("best_action_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = Compose([
    UniformTemporalSubsample(FRAMES),
    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
])

def preprocess_video(frames):
    # Convert & Resize
    frames_resized = [cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (SIZE, SIZE)) for frame in frames]

    clip = np.stack(frames_resized)  # (T, H, W, C)
    clip = torch.from_numpy(clip).float().permute(3, 0, 1, 2)  # (C, T, H, W)

    clip = transform(clip)
    return clip.unsqueeze(0).to(DEVICE)

def predict_action(frames):
    clip = preprocess_video(frames)
    with torch.no_grad():
        logits = model(clip)
        pred = torch.argmax(logits, dim=1).item()
    return CLASS_NAMES[pred]
