import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv("checkpoints/loss_log.csv")

plt.figure(figsize=(8, 5))
plt.plot(log["Epoch"], log["Train Loss"], label="Train Loss", marker='o')
plt.plot(log["Epoch"], log["Validation Loss"], label="Validation Loss", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
