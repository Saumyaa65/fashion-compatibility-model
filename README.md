# 👗 Fashion Compatibility Detection with ResNet + BiLSTM

This project detects whether a group of fashion items (like tops, bottoms, shoes, etc.) go well together. It uses a **ResNet-18** model to extract features from item images and a **BiLSTM** to understand the compatibility between the items in an outfit.

---

## 🧠 What It Does

Given a list of fashion item images that form an outfit, the model:

- Extracts visual features for each item using a pretrained ResNet-18.
- Feeds those features into a BiLSTM to learn the compatibility across the outfit.
- Outputs a single feature vector per outfit for further tasks like:
  - Ranking outfit compatibility
  - Classifying outfits as good/bad
  - Generating outfit recommendations

---

## 🏗️ Architecture

Image → ResNet18 → Feature Embedding → BiLSTM → Outfit Embedding


- **ResNet-18**: Modified to act as a fixed feature extractor (no classification head).
- **BiLSTM**: Models relationships across item features in an outfit.
- **Final output**: A 512D vector representing the whole outfit.

---

## 🗂️ Project Structure

fashion-compatibility/
├── data/
│ └── polyvore_outfits/
│ ├── disjoint/
│ │ └── train.json
│ └── images/
│ └── *.jpg
├── dataset/
│ └── polyvore_dataset.py # Dataset class for loading outfits
├── models/
│ ├── resnet18_model.py # Custom ResNet18 feature extractor
│ ├── resnet_encoder.py # ResNet wrapped as a module
│ └── fashion_encoder.py # ResNet + BiLSTM encoder
├── main.py # Script to test feature extraction
└── README.md


---

## 🛠️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/fashion-compatibility.git
cd fashion-compatibility

### 2. Install requirements
3. Prepare Dataset
Download and extract the Polyvore Outfit Dataset.

Place:

Outfit JSON files in data/polyvore_outfits/disjoint/

Images in data/polyvore_outfits/images/

4. Run Example
python main.py

🧪 Output
Once you run main.py, it will:

Load a batch of outfits from the dataset

Pass each item through the ResNet model

Print shape of feature embeddings from the final layer (expected: [num_items, 512])

📁 Dataset Format
JSON files like train.json contain outfit information (list of item image filenames).

images/ folder contains the corresponding fashion item images.

Each outfit is a list of 2–8 items.

📊 Roadmap
✅ ResNet18 feature extractor

✅ Custom Dataset & Dataloader for Polyvore

✅ Feature generation for outfit items

⏳ BiLSTM outfit encoder (WIP)

⏳ Contrastive Loss training

⏳ Final compatibility scoring module

👩‍💻 Author
Saumya
BTech CSE-AI + Robotics @ VIT Chennai
@yourusername

🙌 Acknowledgements
PyTorch

Polyvore Outfits Dataset

Original ResNet paper: "Deep Residual Learning for Image Recognition"

