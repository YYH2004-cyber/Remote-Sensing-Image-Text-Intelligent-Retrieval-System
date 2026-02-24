import json
from pathlib import Path

import faiss
import numpy as np
import open_clip
import streamlit as st
import torch
from PIL import Image
import os

IMAGES_DIR = Path("/home/dhm/python_project/demo/rsicd_imgs")
INDEX_PATH = Path("/home/dhm/python_project/demo/data/faiss_index.bin")
META_PATH = Path("/home/dhm/python_project/demo/data/meta.json")

TEXT_DIR = Path("./tools/search_queries_with_prefixes.txt")
INDEX_TEXT_PATH = Path("/home/dhm/python_project/demo/data/faiss_index_text.bin")
META_TEXT_PATH = Path("/home/dhm/python_project/demo/data/meta_text.json")

MODEL_WEIGHTS = "/home/dhm/python_project/demo/data/RS5M_ViT-B-32_RSICD.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_history(HISTORY_FILE):
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history, HISTORY_FILE):
    
    """保存历史记录到文件"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, default=str)

def add_to_history(record, HISTORY_FILE):
    """添加新记录到历史"""
    history = load_history(HISTORY_FILE)
    
    # 限制历史记录数量（保留最近50条）
    if len(history) > 50:
        history = history[-49:]
    
    history.append(record)
    save_history(history, HISTORY_FILE)
    return history
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    ckpt = torch.load(
        MODEL_WEIGHTS, map_location=DEVICE
    )
    model.load_state_dict(ckpt)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model.to(DEVICE)
    model.eval()

    return model, preprocess, tokenizer


model, preprocess, tokenizer = load_model()


def image_to_embedding(pil_image: Image.Image):
    image_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(image_tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def text_to_embedding(text: str):
    text_tokens = tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_text(text_tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def build_index(rebuild=False):
    imgs = (
        list(IMAGES_DIR.glob("*.jpg"))
        + list(IMAGES_DIR.glob("*.jpeg"))
        + list(IMAGES_DIR.glob("*.png"))
        + list(IMAGES_DIR.glob("*.tif"))
        + list(IMAGES_DIR.glob("*.tiff"))
    )
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.load(open(META_PATH, "r", encoding="utf-8"))

    text_index = faiss.read_index(str(INDEX_TEXT_PATH))
    meta_text = json.load(open(META_TEXT_PATH, "r", encoding="utf-8"))
    return index, meta, text_index, meta_text
