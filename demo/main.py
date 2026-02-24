import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from utils import (
    IMAGES_DIR,
    add_to_history,
    build_index,
    image_to_embedding,
    load_history,
    save_history,
    text_to_embedding,
)

HISTORY_FILE = Path("./data/search_history.json")

if "history" not in st.session_state:
    st.session_state.history = load_history(HISTORY_FILE)
st.session_state.TOPK = 5


st.set_page_config(
    page_title="🔍 遥感图像-文本智能检索系统",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.header("🔍 遥感图像-文本智能检索系统")
# -----------------------------
# 初始化状态
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "文本 → 图像检索"

# -----------------------------
# 布局：左右两栏
# -----------------------------
left, right = st.columns([0.4, 1.5])  # 里头是两列的宽度比例

# =============================
# 左侧：选择卡（导航）
# =============================
with left:
    st.markdown("#### 🔧 功能选择")

    pages = [
        "文本 → 图像检索",
        "图像 → 图像检索",
        "图像 → 文本描述",
        "历史记录",
    ]

    # radio 本质是“始终可见的选择卡”
    selected = st.radio(
        label="", options=pages, index=pages.index(st.session_state.page)
    )

    # 同步状态
    st.session_state.page = selected


# =============================
# 右侧：页面内容
# =============================
with right:

    with st.spinner("加载模型中...", show_time=True, width="stretch"):
        index, meta, text_index, meta_text = build_index()

    # 每一个页面一个 container
    if st.session_state.page == "文本 → 图像检索":
        with st.container():
            st.subheader("📝 文本 → 图像")
            query = st.text_input("请输入检索文本")

            st.subheader("⚙️ TOPK 设置")
            st.session_state.TOPK = st.select_slider("Top-K", options=[5, 10, 15, 20])
            if st.button("检索图像", key="text2img"):
                if index is None:
                    st.error("没有索引，请先放图片到 ./images 并重启")
                else:
                    q_emb = text_to_embedding(query)
                    D, I = index.search(q_emb, st.session_state.TOPK)
                    st.write(f"Top-{st.session_state.TOPK} 图片结果：")

                    MAX_COLS = 5  # 每行最多显示 5 张图

                    results = list(zip(D[0], I[0]))

                    for row_start in range(0, len(results), MAX_COLS):
                        row_items = results[row_start : row_start + MAX_COLS]
                        cols = st.columns(len(row_items), gap="medium")

                        for col, (score, idx) in zip(cols, row_items):
                            if idx < 0:
                                continue
                            filename = meta[idx]["filename"]
                            img_path = IMAGES_DIR / filename

                            with col:
                                st.image(
                                    str(img_path),
                                    caption=f"{filename}\nscore={score:.4f}",
                                    use_container_width=True,  # ⭐关键
                                )
                    # 添加历史记录（在检索完成后）
                    add_to_history(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "type": "文本 → 图像",
                            "query": query,
                            "topk": st.session_state.TOPK,
                            "results": [
                                meta[idx]["filename"] for idx in I[0] if idx >= 0
                            ],
                        },
                        HISTORY_FILE,
                    )
                    st.session_state.history = load_history(
                        HISTORY_FILE
                    )  # 刷新会话状态

    elif st.session_state.page == "图像 → 图像检索":
        with st.container():
            st.subheader("🖼️ 图像 → 图像")
            st.subheader("上传图像，在图库中检索相似图片")

            st.subheader("⚙️ TOPK 设置")
            st.session_state.TOPK = st.select_slider("Top-K", options=[5, 10, 15, 20])

            uploaded = st.file_uploader(
                "上传一张图片", type=["jpg", "jpeg", "png", "tif", "tiff"], key="i2i"
            )
            if uploaded:
                try:
                    pil = Image.open(uploaded).convert("RGB")
                    st.image(
                        pil,
                        caption="查询图片",
                    )

                    if index is None:
                        st.error("没有索引，请先放图片到 ./images 并重启")
                    else:
                        q_emb = image_to_embedding(pil)
                        D, I = index.search(q_emb, st.session_state.TOPK)
                        st.write(f"Top-{st.session_state.TOPK} 相似图片：")
                        MAX_COLS = 5  # 每行最多显示 5 张图

                    results = list(zip(D[0], I[0]))

                    for row_start in range(0, len(results), MAX_COLS):
                        row_items = results[row_start : row_start + MAX_COLS]
                        cols = st.columns(len(row_items), gap="medium")

                        for col, (score, idx) in zip(cols, row_items):
                            if idx < 0:
                                continue
                            filename = meta[idx]["filename"]
                            img_path = IMAGES_DIR / filename

                            with col:
                                st.image(
                                    str(img_path),
                                    caption=f"{filename}\nscore={score:.4f}",
                                    use_container_width=True,  # ⭐关键
                                )
                    # 添加历史记录（在检索完成后）
                    add_to_history(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "type": "图片 → 图像",
                            "query": uploaded.name,
                            "topk": st.session_state.TOPK,
                            "results": [
                                meta[idx]["filename"] for idx in I[0] if idx >= 0
                            ],
                        },
                        HISTORY_FILE,
                    )
                    st.session_state.history = load_history(
                        HISTORY_FILE
                    )  # 刷新会话状态
                except Exception as e:
                    st.error(f"处理上传的图片时出错: {e}")

    elif st.session_state.page == "图像 → 文本描述":
        with st.container():
            st.subheader("🧠 图像描述")
            uploaded = st.file_uploader(
                "上传一张图片", type=["jpg", "jpeg", "png", "tif", "tiff"], key="i2t"
            )
            st.subheader("⚙️ TOPK 设置")
            st.session_state.TOPK = st.select_slider("Top-K", options=[1, 2, 3, 4, 5])

            if uploaded:
                try:
                    pil = Image.open(uploaded).convert("RGB")
                    st.image(pil, caption="查询图片", width=250)

                    if index is None:
                        st.error("没有索引，请先放图片到 ./images 并重启")
                    else:
                        q_emb = image_to_embedding(pil)
                        D, I = text_index.search(q_emb, st.session_state.TOPK)
                        st.write(f"Top-{st.session_state.TOPK} 文本描述：")
                        cols = st.columns(st.session_state.TOPK, gap="medium")
                        for i, (score, idx) in enumerate(zip(D[0], I[0])):
                            caption = meta_text[idx]["filename"]
                            if idx < 0:
                                continue
                            st.markdown(f"{caption}")
                            st.markdown("-" * 20)
                        add_to_history(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "type": "图像 → 文本",
                                "query": uploaded.name,
                                "topk": st.session_state.TOPK,
                                "results": [
                                    meta_text[idx]["filename"]
                                    for idx in I[0]
                                    if idx >= 0
                                ],
                            },
                            HISTORY_FILE,
                        )
                    st.session_state.history = load_history(
                        HISTORY_FILE
                    )  # 刷新会话状态
                except Exception as e:
                    st.error(f"处理上传的图片时出错: {e}")

    elif st.session_state.page == "历史记录":
        with st.container():
            st.subheader("📜 检索历史记录")

            if not st.session_state.history:
                st.info("暂无历史记录。进行一些检索操作后，记录会显示在这里。")
            else:
                # 排序：最新记录在最前
                sorted_history = sorted(
                    st.session_state.history, key=lambda x: x["timestamp"], reverse=True
                )

                for i, record in enumerate(sorted_history):
                    timestamp = datetime.fromisoformat(record["timestamp"])
                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                    with st.expander(
                        f"🕒 {formatted_time} | {record['type']} | TOP-{record['topk']}"
                    ):
                        # 查询内容
                        st.markdown(f"**查询内容**: `{record['query']}`")

                        # 结果展示
                        st.markdown("**检索结果**:")

                        if record["type"] == "图像→文本":
                            # 文本结果
                            for j, text in enumerate(
                                record["results"][: record["topk"]]
                            ):
                                st.markdown(f"{j + 1}. {text}")
                        else:
                            # 图像结果
                            MAX_COLS = 5
                            results = record["results"][: record["topk"]]

                            for row_start in range(0, len(results), MAX_COLS):
                                row_items = results[row_start : row_start + MAX_COLS]
                                cols = st.columns(len(row_items), gap="medium")

                                for col, filename in zip(cols, row_items):
                                    img_path = IMAGES_DIR / filename
                                    if img_path.exists():
                                        with col:
                                            st.image(
                                                str(img_path),
                                                caption=filename,
                                                use_container_width=True,
                                            )
                                    else:# text result
                                        st.markdown(filename)

                        # 删除单条记录按钮
                        if st.button(
                            "🗑️ 删除这条记录", key=f"del_{i}_{record['timestamp']}"
                        ):
                            st.session_state.history = [
                                h
                                for h in st.session_state.history
                                if h["timestamp"] != record["timestamp"]
                            ]
                            save_history(st.session_state.history,HISTORY_FILE)
                            st.rerun()

                # 清除所有记录按钮
                if st.button("🧹 清除所有历史记录", type="secondary"):
                    if os.path.exists(HISTORY_FILE):
                        os.remove(HISTORY_FILE)
                    st.session_state.history = []
                    st.rerun()
                    st.success("历史记录已清除！")
