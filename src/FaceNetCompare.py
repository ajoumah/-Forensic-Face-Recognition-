pip install facenet-pytorch
import os
import re
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import faiss
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# =============== Config =============== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((160, 160)),  # FaceNet uses 160x160 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # FaceNet normalization
])

# =============== FaceNet Embedding Model =============== #
class FaceNetEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained='vggface2').eval()

    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

# Load model
model = FaceNetEmbedding().to(device)
model.eval()

# =============== Regex Patterns =============== #
pattern_dead = re.compile(r'^personD00(0[1-9]|[1-9][0-9]|1[01][0-9]|12[0-1])\.jpg$', re.IGNORECASE)
pattern_alive = re.compile(r'^PersonA(?:00|a0|aa|ab)(0[1-9]|[1-9][0-9]|1[01][0-9]|12[0-1])\.jpg$', re.IGNORECASE)

# =============== Embedding Function =============== #
def get_embedding(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(tensor)
        return embedding.squeeze().cpu().numpy()
    except Exception as e:
        print(f"❌ Failed to get embedding for {image_path}: {e}")
        return None

# =============== Extract ID Using Regex =============== #
def extract_person_id(filename):
    match_dead = pattern_dead.match(filename)
    match_alive = pattern_alive.match(filename)
    if match_dead:
        return int(match_dead.group(1))
    elif match_alive:
        return int(match_alive.group(1))
    else:
        return None

# =============== Main Comparison Function (Top-5) =============== #
def compare_source_to_destination(source_folder, destination_folder):
    print(f"\n Comparing images:\n📁 Source: {source_folder}\n📁 Destination: {destination_folder}")

    destination_embeddings = []
    destination_filenames = []
    destination_ids = []

    for fname in os.listdir(destination_folder):
        fpath = os.path.join(destination_folder, fname)
        if os.path.isfile(fpath):
            person_id = extract_person_id(fname)
            if person_id is None:
                continue
            emb = get_embedding(fpath)
            if emb is not None:
                destination_embeddings.append(emb.astype(np.float32))
                destination_filenames.append(fname)
                destination_ids.append(person_id)

    if len(destination_embeddings) == 0:
        print("❌ No valid destination embeddings found.")
        return

    d = len(destination_embeddings[0])
    index = faiss.IndexFlatIP(d)
    destination_embeddings_np = np.array(destination_embeddings)
    faiss.normalize_L2(destination_embeddings_np)
    index.add(destination_embeddings_np)

    top1_matches = 0
    top5_matches = 0
    total_compared = 0

    for fname in os.listdir(source_folder):
        fpath = os.path.join(source_folder, fname)
        if os.path.isfile(fpath):
            person_id = extract_person_id(fname)
            if person_id is None:
                continue
            emb = get_embedding(fpath)
            print(f"\n Source: {fname}")
            print(f"Source ID: {person_id}")
            if emb is not None:
                emb = emb.astype(np.float32).reshape(1, -1)
                faiss.normalize_L2(emb)
                D, I = index.search(emb, 5)

                matched = False
                for rank, idx in enumerate(I[0]):
                    if destination_ids[idx] == person_id:
                        print("✅ Matched ID:", destination_ids[idx])
                        if rank == 0:
                            top1_matches += 1
                        top5_matches += 1
                        matched = True
                        break

                if not matched:
                    print(f"❌ No match for: {fname}")

                total_compared += 1

    print("\n Top-K Evaluation Results:")
    print(f" Total Compared: {total_compared}")
    print(f" Top-1 Accuracy: {top1_matches / total_compared:.2%}")
    print(f" Top-5 Accuracy: {top5_matches / total_compared:.2%}")

