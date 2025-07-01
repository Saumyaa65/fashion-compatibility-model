import torch
from main import FashionCompatibilityModel, test, test_loader  # import properly
model = FashionCompatibilityModel().to("cpu")  # or "cuda" if available
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
test(model, test_loader)
