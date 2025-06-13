import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.polyvore_dataset import PolyvoreDataset
from models.resnet_encoder import ResNetEncoder
from models.lstm_model import OutfitLSTM

# -----------------------------
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -----------------------------
# Dataset + Dataloader
dataset = PolyvoreDataset(
    json_path="data/polyvore_outfits/disjoint/train.json",
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



loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# Run one batch
for images, labels in loader:
    images = images.to(device)
    labels = labels.float().to(device)

    outputs = model(images).squeeze(1)  # [B]
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    break

