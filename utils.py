import json
from pathlib import Path

import faiss
import numpy as np
import open_clip
import streamlit as st
import torch
from PIL import Image
import os   

IMAGES_DIR = Path("/home/yyh2004/demo/demo/rsicd_imgs")
INDEX_PATH = Path("/home/yyh2004/demo/demo/data/faiss_index.bin")   
META_PATH = Path("/home/yyh2004/demo/demo/data/meta.json")

TEXT_DIR = Path("/home/yyh2004/demo/demo/tools/dataset_RSITMD.json")
INDEX_TEXT_PATH = Path("/home/yyh2004/demo/demo/data/faiss_index_text.bin")
META_TEXT_PATH = Path("/home/yyh2004/demo/demo/data/meta_text.json")

MODEL_WEIGHTS = "/home/yyh2004/demo/demo/data/RS5M_ViT-B-32_RSICD.pt"
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
        "ViT-B-32", pretrained=None
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


ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024


def validate_uploaded_file(uploaded):
    if uploaded.size > MAX_FILE_SIZE:
        return False, f"文件过大，最大支持{MAX_FILE_SIZE/1024/1024}MB"
    
    ext = Path(uploaded.name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"不支持的文件类型: {ext}"
    
    try:
        img = Image.open(uploaded)
        img.verify()
    except Exception as e:
        return False, f"无效的图像文件: {e}"
    
    return True, "验证通过"


def add_images_to_dataset(uploaded_files):
    added_count = 0
    failed_files = []
    
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    
    try:
        index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except:
        index = faiss.IndexFlatIP(512)
        meta = []
    
    for uploaded in uploaded_files:
        is_valid, message = validate_uploaded_file(uploaded)
        if not is_valid:
            failed_files.append((uploaded.name, message))
            continue
        
        img_path = IMAGES_DIR / uploaded.name
        if img_path.exists():
            failed_files.append((uploaded.name, "文件已存在"))
            continue
        
        try:
            with open(img_path, 'wb') as f:
                f.write(uploaded.getbuffer())
            
            uploaded.seek(0)
            pil = Image.open(uploaded).convert("RGB")
            emb = image_to_embedding(pil)
            
            index.add(emb)
            meta.append({
                "id": len(meta),
                "filename": uploaded.name,
                "size": pil.size
            })
            
            added_count += 1
        except Exception as e:
            failed_files.append((uploaded.name, str(e)))
            if img_path.exists():
                img_path.unlink()
    
    if added_count > 0:
        os.makedirs(INDEX_PATH.parent, exist_ok=True)
        faiss.write_index(index, str(INDEX_PATH))
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    
    return added_count, failed_files


def remove_image_from_dataset(filename):
    img_path = IMAGES_DIR / filename
    
    if not img_path.exists():
        return False, "文件不存在"
    
    try:
        img_path.unlink()
        return True, "删除成功"
    except Exception as e:
        return False, f"删除失败: {e}"


def rebuild_index():
    imgs = (
        list(IMAGES_DIR.glob("*.jpg"))
        + list(IMAGES_DIR.glob("*.jpeg"))
        + list(IMAGES_DIR.glob("*.png"))
        + list(IMAGES_DIR.glob("*.tif"))
        + list(IMAGES_DIR.glob("*.tiff"))
    )
    
    if not imgs:
        return 0
    
    index = faiss.IndexFlatIP(512)
    meta = []
    
    for i, img_path in enumerate(imgs):
        try:
            pil = Image.open(img_path).convert("RGB")
            emb = image_to_embedding(pil)
            index.add(emb)
            meta.append({
                "id": i,
                "filename": img_path.name,
                "size": pil.size
            })
        except Exception as e:
            continue
    
    os.makedirs(INDEX_PATH.parent, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    return len(meta)


def get_dataset_stats():
    imgs = (
        list(IMAGES_DIR.glob("*.jpg"))
        + list(IMAGES_DIR.glob("*.jpeg"))
        + list(IMAGES_DIR.glob("*.png"))
        + list(IMAGES_DIR.glob("*.tif"))
        + list(IMAGES_DIR.glob("*.tiff"))
    )
    
    total_size = sum(img.stat().st_size for img in imgs)
    format_counts = {}
    
    for img in imgs:
        ext = img.suffix.lower()
        format_counts[ext] = format_counts.get(ext, 0) + 1
    
    return {
        "total_images": len(imgs),
        "total_size": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "format_counts": format_counts
    }


def get_all_images(page=1, per_page=20):
    imgs = (
        list(IMAGES_DIR.glob("*.jpg"))
        + list(IMAGES_DIR.glob("*.jpeg"))
        + list(IMAGES_DIR.glob("*.png"))
        + list(IMAGES_DIR.glob("*.tif"))
        + list(IMAGES_DIR.glob("*.tiff"))
    )
    
    imgs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    total = len(imgs)
    start = (page - 1) * per_page
    end = start + per_page
    
    return imgs[start:end], total
