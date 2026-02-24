import json
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image

IMAGES_DIR = Path("/home/dhm/python_project/demo/rsicd_imgs")
INDEX_PATH = Path("faiss_index.bin")
META_PATH = Path("meta.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")

    return model, preprocess, tokenizer


def image_to_embedding(model, preprocess, pil_image: Image.Image):
    image_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(image_tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def build_index():
    img_extensions = ["*.jpg", "*.png", "*.jpeg", "*.tif", "*.tiff"]
    imgs = []
    for ext in img_extensions:
        imgs.extend(list(IMAGES_DIR.glob(ext)))

    if len(imgs) == 0:
        print("⚠️ 找不到图片 请把图片添加到 ./images 文件夹下")
        return None, []

    print(f"Found {len(imgs)} images. Building index...")

    model, preprocess, _ = load_model()

    embeddings, meta = [], []
    for i, p in enumerate(imgs):
        try:
            pil = Image.open(p).convert("RGB")
            emb = image_to_embedding(model, preprocess, pil)
            embeddings.append(emb[0])
            meta.append({"id": i, "filename": p.name})
            print(f"正在处理 {i + 1}/{len(imgs)}: {p.name}")
        except Exception as e:
            print(f"处理错误 {p.name}: {e}")
            continue

    if not embeddings:
        print("No images were successfully processed")
        return None, []

    xb = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Index built successfully with {len(meta)} images")
    print(f"Index saved to {INDEX_PATH}")
    print(f"Metadata saved to {META_PATH}")

    return index, meta


if __name__ == "__main__":
    build_index()
