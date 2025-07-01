import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.polyvore_dataset import PolyvoreDataset
from models.resnet_encoder import ResNetEncoder
from models.lstm_model import OutfitLSTM

from torch.utils.data import Subset
import random
import os
import csv
import numpy as np


os.makedirs("checkpoints", exist_ok=True)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# -----------------------------
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),   # randomly flip outfit images
    transforms.RandomRotation(10),            # small random rotation
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# -----------------------------
# Dataset + Dataloader
train_dataset = PolyvoreDataset(
    json_path="data/polyvore_outfits/disjoint/combined_train.json",
    image_dir="data/polyvore_outfits/images",
    transform=transform
)
valid_dataset = PolyvoreDataset(
    json_path="data/polyvore_outfits/disjoint/combined_valid.json",
    image_dir="data/polyvore_outfits/images",
    transform=transform
)
test_dataset = PolyvoreDataset(
    json_path="data/polyvore_outfits/disjoint/combined_test.json",
    image_dir="data/polyvore_outfits/images",
    transform=transform
)

def custom_collate_fn(batch):
    outfits, labels = zip(*batch)
    max_len = max(len(o) for o in outfits)

    padded_outfits = []
    for outfit in outfits:
        # If outfit is already a Tensor, convert it to list
        if isinstance(outfit, torch.Tensor):
            outfit = list(outfit)

        pad_len = max_len - len(outfit)
        if pad_len > 0:
            padding = [torch.zeros_like(outfit[0]) for _ in range(pad_len)]
            outfit = outfit + padding

        padded_outfits.append(torch.stack(outfit))  # [N, 3, H, W]

    outfit_tensors = torch.stack(padded_outfits)  # [B, N, 3, H, W]
    return outfit_tensors, torch.tensor(labels)
    

num_samples = int(0.03 * len(train_dataset))  # take 10% data
indices = random.sample(range(len(train_dataset)), num_samples)
small_train_dataset = Subset(train_dataset, indices)

num_samples = int(0.03 * len(valid_dataset))  # take 10% data
indices = random.sample(range(len(valid_dataset)), num_samples)
small_valid_dataset = Subset(valid_dataset, indices)

num_samples = int(0.03 * len(test_dataset))  # take 10% data
indices = random.sample(range(len(test_dataset)), num_samples)
small_test_dataset = Subset(test_dataset, indices)

train_loader = DataLoader(small_train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
valid_loader = DataLoader(small_valid_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
test_loader = DataLoader(small_test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)


# -----------------------------
# Model setup
class FashionCompatibilityModel(nn.Module):
    def __init__(self):
        super(FashionCompatibilityModel, self).__init__()
        self.encoder = ResNetEncoder()
        self.lstm = OutfitLSTM()

    def forward(self, outfit_images):  
        B, N, C, H, W = outfit_images.shape
        outfit_images = outfit_images.view(B * N, C, H, W)
        features = self.encoder(outfit_images)  # [B*N, 512]
        features = features.view(B, N, -1)      # [B, N, 512]
        return self.lstm(features)              # [B, 1]

model = FashionCompatibilityModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

@torch.no_grad()
def validate(model, valid_loader):
    model.eval()
    total_loss = 0.0

    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

    avg_val_loss = total_loss / len(valid_loader)
    return avg_val_loss


@torch.no_grad()
def test(model, test_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(test_loader, desc="Testing", leave=False):
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds.float() == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    print(f"\n🧪 Test Loss: {avg_loss:.4f} | Accuracy: {accuracy*100:.2f}%\n")


# -----------------------------
import time

num_epochs = 1  # adjust as needed

if __name__ == "__main__":
    
    print(len(small_train_dataset))
    print(len(small_valid_dataset))
    print(len(small_train_dataset))
    best_val_loss = float('inf')

    with open("checkpoints/loss_log.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_loss:.4f}")

        val_loss = validate(model, valid_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss:.4f}")

        # Save losses to CSV
        with open("checkpoints/loss_log.csv", mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, val_loss])


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("🎯 Best model updated and saved!")
    
    # Load and test best model
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    test(model, test_loader)


        

