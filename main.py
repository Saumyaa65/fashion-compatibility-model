from dataset.polyvore_dataset import PolyvoreDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    # Assuming each batch is a list of tuples (outfit, label)
    outfits, labels = zip(*batch)
    
    # Pad outfits to the same length (if necessary)
    max_length = max(len(outfit) for outfit in outfits)
    padded_outfits = [outfit + [None] * (max_length - len(outfit)) for outfit in outfits]
    
    # Convert padded outfits to tensors (if applicable)
    # For example, if outfits are image paths, you might load and process images here
    
    return padded_outfits, labels


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = PolyvoreDataset(
    json_path="data/polyvore_outfits/disjoint/train.json",
    image_dir="data/polyvore_outfits/images",
    transform=transform
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

for batch in loader:
    outfits, labels = batch
    print(batch)
    for i, outfit in enumerate(outfits):
        print(f"Outfit {i+1}: {len(outfit)} items")
    break


