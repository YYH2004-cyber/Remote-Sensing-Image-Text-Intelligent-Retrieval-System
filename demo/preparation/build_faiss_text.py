import json
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image

TEXT_DIR = Path("/home/dhm/python_project/demo/tools/captions.txt")
INDEX_PATH = Path("faiss_index_text.bin")
META_PATH = Path("meta_text.json")
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


def text_to_embedding(model, tokenizer, sentence):
    text_tensor = tokenizer(sentence).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_text(text_tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def build_index():
    with open('/home/dhm/python_project/demo/tools/captions.txt', 'r') as f:
        text = f.readlines()
    
        print(f"Found {len(text)} sentence. Building index...")

        model, preprocess, tokenizer = load_model()

        embeddings, meta = [], []
        for i, p in enumerate(text):
            try:
                emb = text_to_embedding(model, tokenizer, p)
                embeddings.append(emb[0])
                meta.append({"id": i, "filename": p})
                print(f"正在处理 {i + 1}/{len(text)}: {p}")
            except Exception as e:
                print(f"处理错误 {p}: {e}")
                continue

        if not embeddings:
            print("No text were successfully processed")
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
    # with open('tools/search_queries_with_prefixes.txt', 'r') as f:
    #     text = f.readlines()
    #     test_list = list()
    #     print(len(text))
    #     for i in range(0, 10):
    #         test_list.append(text[i])
    #     tokenizer = open_clip.get_tokenizer("ViT-B-32")
    #     model, _, preprocess = open_clip.create_model_and_transforms(
    #     "ViT-B-32", pretrained="openai"
    #     )
    #     model.to(DEVICE)
    #     for i in range(0, 10):
    #         print(text_to_embedding(model, tokenizer, test_list[i]).shape)

