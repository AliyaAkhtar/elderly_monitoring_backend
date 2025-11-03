import os
import cv2
import numpy as np
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.transforms import Compose
from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import (
    UniformTemporalSubsample,
    Normalize,
    ShortSideScale,
    RandomShortSideScale
)
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo
)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 9
FRAMES = 16
SIZE = 224
BATCH_SIZE = 4  # Reduced for faster iteration
EPOCHS = 20  # Optimal for pretrained models
LEARNING_RATE = 1e-4
PATIENCE = 5  # Early stopping patience

ROOT = r"C:\Users\Hp\Desktop\CVF\NTU_RGBD_Cross_Subject_Arranged_Dataset_for_Features_File_Creation_(Splitted)"
CLASS_IDS = [41,42,43,44,45,46,47,48,49]

CLASS_NAMES = {
    41: "sneeze/cough",
    42: "staggering",
    43: "falling",
    44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea/vomiting",
    49: "use a fan / feeling warm"
}

# ---------- OPTIMIZED VIDEO LOADER ----------
def load_video(path, num_frames=FRAMES):
    """Load and sample video frames efficiently"""
    cap = cv2.VideoCapture(path)
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # Sample frame indices uniformly
    if total_frames < num_frames:
        indices = np.arange(total_frames).tolist() * (num_frames // total_frames + 1)
        indices = indices[:num_frames]
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    frames = []
    current_idx = 0
    
    for target_idx in indices:
        # Skip frames efficiently
        while current_idx < target_idx:
            cap.grab()
            current_idx += 1
        
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize immediately to save memory
            frame = cv2.resize(frame, (SIZE, SIZE))
            frames.append(frame)
        current_idx += 1
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # Pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    return np.stack(frames[:num_frames])


# ---------- DATASET ----------
class VideoDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform
        self.cid_to_label = {cid: i for i, cid in enumerate(CLASS_IDS)}

    def __getitem__(self, i):
        path, cid = self.items[i]
        clip = load_video(path)
        
        if clip is None:
            clip = np.zeros((FRAMES, SIZE, SIZE, 3), dtype=np.uint8)
        
        clip = torch.from_numpy(clip).float()
        clip = clip.permute(3,0,1,2)  # (C,T,H,W)
        clip = self.transform(clip)
        
        return clip, self.cid_to_label[cid]

    def __len__(self):
        return len(self.items)


# ---------- DATA DISCOVERY ----------
def discover():
    items = []
    for cid in CLASS_IDS:
        for path in glob(os.path.join(ROOT, str(cid), "**/*.avi"), recursive=True):
            subject = os.path.basename(os.path.dirname(path))
            items.append((path, cid, subject))
    return items


def prepare_data():
    all_items = discover()
    
    TRAIN = {f"P{idx:03d}" for idx in range(1, 33)}
    VAL   = {f"P{idx:03d}" for idx in range(33, 37)}
    TEST  = {f"P{idx:03d}" for idx in range(37, 41)}
    
    train_items, val_items, test_items = [], [], []
    for p, cid, s in all_items:
        if s in TRAIN: train_items.append((p, cid))
        elif s in VAL: val_items.append((p, cid))
        elif s in TEST: test_items.append((p, cid))
        else: train_items.append((p, cid))
    
    return train_items, val_items, test_items


def get_transforms():
    train_transform = Compose([
        UniformTemporalSubsample(FRAMES),
        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
    ])
    
    val_transform = Compose([
        UniformTemporalSubsample(FRAMES),
        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
    ])
    
    return train_transform, val_transform


def train_model():
    # Prepare data
    print("🔍 Discovering videos...")
    train_items, val_items, test_items = prepare_data()
    print(f"📊 Dataset Split:")
    print(f"  Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    print("📦 Creating datasets...")
    train_ds = VideoDataset(train_items, train_transform)
    val_ds = VideoDataset(val_items, val_transform)
    test_ds = VideoDataset(test_items, val_transform)
    
    # Create data loaders
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=0, pin_memory=True)
    
    # Load model
    print(f"🤖 Loading X3D-M Model...")
    model = x3d_m(pretrained=True)
    model.blocks[5].proj = nn.Linear(model.blocks[5].proj.in_features, NUM_CLASSES)
    model = model.to(DEVICE)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print(f"🚀 Starting Training on {DEVICE}...\n")
    
    best_val_acc = 0.0
    train_losses, val_accs = [], []
    patience_counter = 0
    
    # Training loop
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for clips, labels in pbar:
            clips = clips.to(DEVICE)
            labels = labels.to(DEVICE)
            
            preds = model(clips)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            correct_train += (preds.argmax(1) == labels).sum().item()
            total_train += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{correct_train/total_train:.3f}'})
        
        avg_train_loss = train_loss / len(train_dl)
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        correct = total = 0
        
        with torch.no_grad():
            for clips, labels in tqdm(val_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  "):
                clips = clips.to(DEVICE)
                labels = labels.to(DEVICE)
                
                preds = model(clips).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        val_accs.append(val_acc)
        scheduler.step()
        
        # Save best model and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_action_model.pth')
            print(f"✅ Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {avg_train_loss:.4f} ⭐ BEST\n")
        else:
            patience_counter += 1
            print(f"   Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {avg_train_loss:.4f}\n")
            
            if patience_counter >= PATIENCE:
                print(f"⚠️  Early stopping triggered! No improvement for {PATIENCE} epochs.")
                print(f"   Best Val Acc: {best_val_acc:.4f} at epoch {epoch+1-PATIENCE}")
                break
    
    # Load best model
    print(f"\n📥 Loading best model (Val Acc: {best_val_acc:.4f})...")
    model.load_state_dict(torch.load('best_action_model.pth'))
    
    # Testing
    print("\n🧪 Testing on Test Set...")
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for clips, labels in tqdm(test_dl, desc="Testing"):
            clips = clips.to(DEVICE)
            preds = model(clips).argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
    
    test_acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
    
    print(f"\n{'='*60}")
    print(f"✅ FINAL TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'='*60}\n")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(CLASS_NAMES.values())))
    
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("\n📈 Training curves saved to 'training_curves.png'")


if __name__ == '__main__':
    train_model()