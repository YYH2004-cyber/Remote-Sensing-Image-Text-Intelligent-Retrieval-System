import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from utils import (
    IMAGES_DIR,
    add_to_history,
    add_images_to_dataset,
    build_index,
    get_all_images,
    get_dataset_stats,
    image_to_embedding,
    load_history,
    rebuild_index,
    remove_image_from_dataset,
    save_history,
    text_to_embedding,
)

HISTORY_FILE = Path("./demo/data/search_history.json")

if "history" not in st.session_state:
    st.session_state.history = load_history(HISTORY_FILE)
st.session_state.TOPK = 5

st.set_page_config(
    page_title="🔍 遥感图像-文本智能检索系统",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* 全局样式 - 纯白色背景 */
    div[data-testid="stApp"] {
        background: #FFFFFF;
        position: relative;
        min-height: 100vh;
    }
    
    /* 移除所有伪元素背景效果 */
    div[data-testid="stApp"]::before {
        display: none;
    }
    
    div[data-testid="stApp"]::after {
        display: none;
    }
    
    /* 移除动画 */
    @keyframes gradientShift {
        0%, 100% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
            opacity: 1;
        }
        50% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    /* 主容器样式 - 适应白色背景 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: #FFFFFF;
        border-radius: 20px;
        box-shadow: 
            0 4px 6px rgba(0, 0, 0, 0.05),
            0 10px 15px rgba(0, 0, 0, 0.1),
            0 0 0 1px rgba(0, 0, 0, 0.05);
        margin-top: 1rem;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    /* 标题样式 */
    h1 {
        color: #1a202c !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        text-align: center;
        margin-bottom: 1.5rem !important;
        letter-spacing: -0.02em;
    }
    
    /* 副标题样式 */
    h2 {
        color: #2d3748 !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        letter-spacing: -0.01em;
    }
    
    h3 {
        color: #2d3748 !important;
        font-weight: 600 !important;
        font-size: 1.4rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* 文本样式 */
    .stMarkdown {
        color: #4a5568;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* 次要按钮样式 */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);
    }
    
    /* 输入框样式 */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
        outline: none;
    }
    
    /* 文件上传器样式 */
    .stFileUploader {
        border: 2px dashed #cbd5e0;
        border-radius: 15px;
        padding: 2rem;
        background: #f7fafc;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        background: #edf2f7;
        border-color: #667eea;
    }
    
    /* 滑块样式 */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Radio按钮样式 */
    .stRadio > div {
        background: #f7fafc;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .stRadio label {
        color: #2d3748;
        font-weight: 500;
        font-size: 1.1rem;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stRadio label:hover {
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Expander样式 */
    .streamlit-expanderHeader {
        background: #f7fafc;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #edf2f7;
        border-color: #667eea;
    }
    
    /* 图片容器样式 */
    img {
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    img:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* 信息框样式 */
    .stInfo {
        background: #ebf8ff;
        border-left: 4px solid #4299e1;
        border-radius: 10px;
        padding: 1.5rem;
        color: #2b6cb0;
    }
    
    /* 错误框样式 */
    .stError {
        background: #fff5f5;
        border-left: 4px solid #f56565;
        border-radius: 10px;
        padding: 1.5rem;
        color: #c53030;
    }
    
    /* 成功框样式 */
    .stSuccess {
        background: #f0fff4;
        border-left: 4px solid #48bb78;
        border-radius: 10px;
        padding: 1.5rem;
        color: #276749;
    }
    
    /* 警告框样式 */
    .stWarning {
        background: #fffff0;
        border-left: 4px solid #ecc94b;
        border-radius: 10px;
        padding: 1.5rem;
        color: #975a16;
    }
    
    /* 加载动画 */
    .stSpinner {
        color: #667eea;
    }
    
    /* 列间距 */
    div[data-testid="stHorizontalBlock"] > div {
        gap: 1.5rem;
    }
    
    /* 标记文本样式 */
    .stMarkdown {
        color: #2d3748;
        line-height: 1.6;
    }
    
    /* 代码块样式 */
    .stCode {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-family: 'Courier New', monospace;
        color: #667eea;
    }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background: #f7fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* 侧边栏内容样式 */
    [data-testid="stSidebar"] .stMarkdown {
        color: #2d3748;
    }
    
    /* 动画效果 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* 应用动画 */
    .main .block-container {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* 响应式设计 - 优化布局 */
    @media (max-width: 1200px) {
        .main .block-container {
            max-width: 1100px;
            margin-left: auto;
            margin-right: auto;
        }
    }
    
    @media (max-width: 768px) {
        h1 { font-size: 2rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.2rem !important; }
        .stButton > button { font-size: 1rem; padding: 0.6rem 1.5rem; }
        
        .main .block-container {
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            border-radius: 15px;
        }
    }
    
    @media (max-width: 480px) {
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
        
        .main .block-container {
            background: #FFFFFF;
            margin-top: 0.3rem;
            margin-bottom: 0.3rem;
            padding-top: 1rem;
            padding-bottom: 1rem;
            border-radius: 12px;
        }
    }
    
    @media (min-width: 1440px) {
        .main .block-container {
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }
    }
    
    /* 深色模式支持 */
    @media (prefers-color-scheme: dark) {
        div[data-testid="stApp"] {
            background: #1a202c;
        }
        
        .main .block-container {
            background: #2d3748;
            box-shadow: 
                0 4px 6px rgba(0, 0, 0, 0.3),
                0 10px 15px rgba(0, 0, 0, 0.4),
                0 0 0 1px rgba(255, 255, 255, 0.1);
        }
        
        h1, h2, h3 {
            color: #f7fafc !important;
        }
        
        .stMarkdown {
            color: #e2e8f0;
        }
        
        .stRadio label {
            color: #e2e8f0;
        }
        
        [data-testid="stSidebar"] {
            background: #2d3748;
            border-right: 1px solid #4a5568;
        }
        
        .stFileUploader {
            background: #2d3748;
            border-color: #4a5568;
        }
        
        .stRadio > div {
            background: #2d3748;
        }
        
        .streamlit-expanderHeader {
            background: #2d3748;
            border-color: #4a5568;
        }
    }
    
    /* 减少动画效果 - 针对性能较低的设备 */
    @media (prefers-reduced-motion: reduce) {
        div[data-testid="stApp"] {
            animation: none;
        }
        
        div[data-testid="stApp"]::after {
            animation: none;
        }
        
        .main .block-container {
            animation: none;
        }
        
        img {
            transition: none;
        }
        
        .stButton > button {
            transition: none;
        }
    }
    
    /* 自定义滚动条 */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🔍 遥感图像-文本智能检索系统</h1>", unsafe_allow_html=True)
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
# 左侧：功能导航面板
# =============================
with left:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        margin-bottom: 1.5rem;
    ">
        <h3 style="
            color: #2d3748;
            margin: 0 0 1.5rem 0;
            font-size: 1.5rem;
            text-align: center;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        ">🔧 功能导航</h3>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        "📝 文本 → 图像检索",
        "🖼️ 图像 → 图像检索",
        "🧠 图像 → 文本描述",
        "📜 历史记录",
        "📊 数据集管理",
    ]

    page_mapping = {
        "文本 → 图像检索": "📝 文本 → 图像检索",
        "图像 → 图像检索": "🖼️ 图像 → 图像检索",
        "图像 → 文本描述": "🧠 图像 → 文本描述",
        "历史记录": "📜 历史记录",
        "数据集管理": "📊 数据集管理"
    }

    current_page_with_icon = page_mapping.get(st.session_state.page, "📝 文本 → 图像检索")

    selected = st.radio(
        label="选择功能",
        options=pages,
        index=pages.index(current_page_with_icon),
        label_visibility="collapsed"
    )

    st.session_state.page = selected.replace("📝 ", "").replace("🖼️ ", "").replace("🧠 ", "").replace("📜 ", "").replace("📊 ", "")


# =============================
# 右侧：功能内容区域
# =============================
with right:
    with st.spinner("加载模型中...", show_time=True, width="stretch"):
        index, meta, text_index, meta_text = build_index()

    if st.session_state.page == "文本 → 图像检索":
        with st.container():
            st.markdown("<h2>📝 文本 → 图像检索</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border-left: 4px solid #667eea;
            ">
                <p style="margin: 0; color: #4a5568; font-size: 1rem;">
                    � <strong>提示：</strong>输入文本描述，系统将检索最相似的遥感图像
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            query = st.text_input(
                "请输入检索文本",
                placeholder="例如：机场跑道、建筑物群、森林覆盖等...",
                label_visibility="visible"
            )

            st.markdown("<h3>⚙️ 参数设置</h3>", unsafe_allow_html=True)
            st.session_state.TOPK = st.select_slider(
                "返回结果数量",
                options=[5, 10, 15, 20],
                value=5,
                format_func=lambda x: f"{x} 张图片"
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔍 开始检索", key="text2img", use_container_width=True):
                    if index is None:
                        st.error("❌ 没有索引，请先放图片到 ./images 并重启")
                    else:
                        q_emb = text_to_embedding(query)
                        D, I = index.search(q_emb, st.session_state.TOPK)
                        
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                            border-radius: 15px;
                            padding: 1.5rem;
                            margin-top: 1.5rem;
                            text-align: center;
                        ">
                            <h3 style="margin: 0; color: #2d3748;">
                                ✨ 检索完成！共找到 {st.session_state.TOPK} 张相关图片
                            </h3>
                        </div>
                        """, unsafe_allow_html=True)

                        MAX_COLS = 5
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
                                        caption=f"📁 {filename}\n🎯 相似度: {score:.4f}",
                                        use_container_width=True,
                                    )
                    
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
                    st.session_state.history = load_history(HISTORY_FILE)

    elif st.session_state.page == "图像 → 图像检索":
        with st.container():
            st.markdown("<h2>🖼️ 图像 → 图像检索</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border-left: 4px solid #667eea;
            ">
                <p style="margin: 0; color: #4a5568; font-size: 1rem;">
                    💡 <strong>提示：</strong>上传一张遥感图像，系统将在图库中检索相似的图片
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<h3>⚙️ 参数设置</h3>", unsafe_allow_html=True)
            st.session_state.TOPK = st.select_slider(
                "返回结果数量",
                options=[5, 10, 15, 20],
                value=5,
                format_func=lambda x: f"{x} 张图片"
            )

            st.markdown("<h3>📤 上传图像</h3>", unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "选择一张图片",
                type=["jpg", "jpeg", "png", "tif", "tiff"],
                key="i2i",
                label_visibility="collapsed"
            )
            
            if uploaded:
                try:
                    pil = Image.open(uploaded).convert("RGB")
                    
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                        border-radius: 15px;
                        padding: 1.5rem;
                        margin-bottom: 1.5rem;
                        text-align: center;
                    ">
                        <h3 style="margin: 0; color: #2d3748;">📸 查询图片</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.image(pil, caption=f"📁 {uploaded.name}", use_container_width=True)

                    if index is None:
                        st.error("❌ 没有索引，请先放图片到 ./images 并重启")
                    else:
                        q_emb = image_to_embedding(pil)
                        D, I = index.search(q_emb, st.session_state.TOPK)
                        
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                            border-radius: 15px;
                            padding: 1.5rem;
                            margin-top: 1.5rem;
                            text-align: center;
                        ">
                            <h3 style="margin: 0; color: #2d3748;">
                                ✨ 检索完成！共找到 {st.session_state.TOPK} 张相似图片
                            </h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        MAX_COLS = 5
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
                                        caption=f"📁 {filename}\n🎯 相似度: {score:.4f}",
                                        use_container_width=True,
                                    )
                    
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
                    st.session_state.history = load_history(HISTORY_FILE)
                except Exception as e:
                    st.error(f"❌ 处理上传的图片时出错: {e}")

    elif st.session_state.page == "图像 → 文本描述":
        with st.container():
            st.markdown("<h2>🧠 图像 → 文本描述</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border-left: 4px solid #667eea;
            ">
                <p style="margin: 0; color: #4a5568; font-size: 1rem;">
                    💡 <strong>提示：</strong>上传一张遥感图像，系统将自动生成文本描述
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<h3>⚙️ 参数设置</h3>", unsafe_allow_html=True)
            st.session_state.TOPK = st.select_slider(
                "返回描述数量",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: f"{x} 条描述"
            )

            st.markdown("<h3>📤 上传图像</h3>", unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "选择一张图片",
                type=["jpg", "jpeg", "png", "tif", "tiff"],
                key="i2t",
                label_visibility="collapsed"
            )

            if uploaded:
                try:
                    pil = Image.open(uploaded).convert("RGB")
                    
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                        border-radius: 15px;
                        padding: 1.5rem;
                        margin-bottom: 1.5rem;
                        text-align: center;
                    ">
                        <h3 style="margin: 0; color: #2d3748;">📸 查询图片</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.image(pil, caption=f"📁 {uploaded.name}", use_container_width=True)

                    if index is None:
                        st.error("❌ 没有索引，请先放图片到 ./images 并重启")
                    else:
                        q_emb = image_to_embedding(pil)
                        D, I = text_index.search(q_emb, st.session_state.TOPK)
                        
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                            border-radius: 15px;
                            padding: 1.5rem;
                            margin-top: 1.5rem;
                            text-align: center;
                        ">
                            <h3 style="margin: 0; color: #2d3748;">
                                ✨ 描述生成完成！共生成 {st.session_state.TOPK} 条文本描述
                            </h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        cols = st.columns(st.session_state.TOPK, gap="medium")
                        for i, (score, idx) in enumerate(zip(D[0], I[0])):
                            caption = meta_text[idx]["filename"]
                            if idx < 0:
                                continue
                            with cols[i]:
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                                    border-radius: 15px;
                                    padding: 1.5rem;
                                    border-left: 4px solid #667eea;
                                ">
                                    <h4 style="margin: 0 0 1rem 0; color: #2d3748;">
                                        📝 描述 {i + 1}
                                    </h4>
                                    <p style="margin: 0; color: #4a5568; line-height: 1.6;">
                                        {caption}
                                    </p>
                                    <p style="margin: 1rem 0 0 0; color: #667eea; font-size: 0.9rem;">
                                        🎯 相似度: {score:.4f}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        
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
                    st.session_state.history = load_history(HISTORY_FILE)
                except Exception as e:
                    st.error(f"❌ 处理上传的图片时出错: {e}")

    elif st.session_state.page == "历史记录":
        with st.container():
            st.markdown("<h2>📜 检索历史记录</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border-left: 4px solid #667eea;
            ">
                <p style="margin: 0; color: #4a5568; font-size: 1rem;">
                    � <strong>提示：</strong>查看和管理您的检索历史记录
                </p>
            </div>
            """, unsafe_allow_html=True)

            if not st.session_state.history:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                    border-radius: 15px;
                    padding: 3rem;
                    text-align: center;
                ">
                    <h3 style="margin: 0 0 1rem 0; color: #2d3748; font-size: 1.5rem;">
                        📭 暂无历史记录
                    </h3>
                    <p style="margin: 0; color: #4a5568; font-size: 1.1rem;">
                        进行一些检索操作后，记录会显示在这里
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                sorted_history = sorted(
                    st.session_state.history, key=lambda x: x["timestamp"], reverse=True
                )

                for i, record in enumerate(sorted_history):
                    timestamp = datetime.fromisoformat(record["timestamp"])
                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                    with st.expander(
                        f"🕒 {formatted_time} | {record['type']} | TOP-{record['topk']}",
                        expanded=False
                    ):
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
                            border-radius: 12px;
                            padding: 1.5rem;
                            margin-bottom: 1rem;
                            border-left: 4px solid #667eea;
                        ">
                            <h4 style="margin: 0 0 0.5rem 0; color: #2d3748; font-size: 1.2rem;">
                                🔍 查询内容
                            </h4>
                            <p style="margin: 0; color: #4a5568; font-family: 'Courier New', monospace; background: rgba(102, 126, 234, 0.1); padding: 0.5rem 1rem; border-radius: 8px;">
                                {record['query']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
                            border-radius: 12px;
                            padding: 1.5rem;
                            margin-bottom: 1rem;
                            border-left: 4px solid #667eea;
                        ">
                            <h4 style="margin: 0 0 1rem 0; color: #2d3748; font-size: 1.2rem;">
                                📊 检索结果
                            </h4>
                        </div>
                        """, unsafe_allow_html=True)

                        if record["type"] == "图像→文本":
                            for j, text in enumerate(record["results"][: record["topk"]]):
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
                                    border-radius: 10px;
                                    padding: 1rem;
                                    margin-bottom: 0.5rem;
                                    border-left: 3px solid #667eea;
                                ">
                                    <p style="margin: 0; color: #4a5568; line-height: 1.6;">
                                        <strong>📝 描述 {j + 1}:</strong> {text}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
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
                                                caption=f"📁 {filename}",
                                                use_container_width=True,
                                            )
                                    else:
                                        st.markdown(f"""
                                        <div style="
                                            background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
                                            border-radius: 10px;
                                            padding: 1rem;
                                            text-align: center;
                                        ">
                                            <p style="margin: 0; color: #4a5568;">
                                                {filename}
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)

                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if st.button(
                                "🗑️ 删除这条记录", 
                                key=f"del_{i}_{record['timestamp']}",
                                use_container_width=True
                            ):
                                st.session_state.history = [
                                    h
                                    for h in st.session_state.history
                                    if h["timestamp"] != record["timestamp"]
                                ]
                                save_history(st.session_state.history, HISTORY_FILE)
                                st.rerun()

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🧹 清除所有历史记录", type="secondary", use_container_width=True):
                        if os.path.exists(HISTORY_FILE):
                            os.remove(HISTORY_FILE)
                        st.session_state.history = []
                        st.rerun()
                        st.success("✅ 历史记录已清除！")

    elif st.session_state.page == "数据集管理":
        with st.container():
            st.markdown("<h2>📊 数据集管理</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border-left: 4px solid #667eea;
            ">
                <p style="margin: 0; color: #4a5568; font-size: 1rem;">
                    💡 <strong>提示：</strong>管理您的图像数据集，包括添加、删除和查看图像
                </p>
            </div>
            """, unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["📈 统计信息", "➕ 添加图像", "🗑️ 删除图像"])

            with tab1:
                st.markdown("<h3>📈 数据集统计</h3>", unsafe_allow_html=True)
                
                stats = get_dataset_stats()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                        border-radius: 15px;
                        padding: 2rem;
                        text-align: center;
                    ">
                        <h4 style="margin: 0 0 1rem 0; color: #2d3748; font-size: 1rem;">图像总数</h4>
                        <p style="margin: 0; color: #667eea; font-size: 2.5rem; font-weight: bold;">
                            {stats['total_images']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                        border-radius: 15px;
                        padding: 2rem;
                        text-align: center;
                    ">
                        <h4 style="margin: 0 0 1rem 0; color: #2d3748; font-size: 1rem;">总大小</h4>
                        <p style="margin: 0; color: #667eea; font-size: 2.5rem; font-weight: bold;">
                            {stats['total_size_mb']:.2f} MB
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                        border-radius: 15px;
                        padding: 2rem;
                        text-align: center;
                    ">
                        <h4 style="margin: 0 0 1rem 0; color: #2d3748; font-size: 1rem;">格式数量</h4>
                        <p style="margin: 0; color: #667eea; font-size: 2.5rem; font-weight: bold;">
                            {len(stats['format_counts'])}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if stats['format_counts']:
                    st.markdown("<h4>📊 格式分布</h4>", unsafe_allow_html=True)
                    for ext, count in stats['format_counts'].items():
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
                            border-radius: 10px;
                            padding: 1rem;
                            margin-bottom: 0.5rem;
                            border-left: 3px solid #667eea;
                        ">
                            <p style="margin: 0; color: #4a5568;">
                                <strong>{ext.upper()}</strong>: {count} 张图片
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

            with tab2:
                st.markdown("<h3>➕ 添加图像</h3>", unsafe_allow_html=True)
                
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                ">
                    <p style="margin: 0; color: #4a5568; line-height: 1.6;">
                        📁 <strong>支持的格式：</strong>JPG, JPEG, PNG, TIF, TIFF<br>
                        📏 <strong>文件大小限制：</strong>最大 10MB<br>
                        💡 <strong>提示：</strong>可以一次上传多个文件
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                uploaded_files = st.file_uploader(
                    "选择图像文件",
                    type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
                    accept_multiple_files=True,
                    help="支持批量上传，最多10MB/文件"
                )
                
                if uploaded_files:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                        border-radius: 12px;
                        padding: 1.5rem;
                        margin-bottom: 1.5rem;
                    ">
                        <p style="margin: 0; color: #4a5568;">
                            📎 已选择 <strong>{len(uploaded_files)}</strong> 个文件
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🚀 开始添加", type="primary", use_container_width=True):
                            with st.spinner("正在处理图像..."):
                                added_count, failed_files = add_images_to_dataset(uploaded_files)
                                
                                if added_count > 0:
                                    st.success(f"✅ 成功添加 {added_count} 张图片！")
                                
                                if failed_files:
                                    st.warning(f"⚠️ {len(failed_files)} 个文件添加失败：")
                                    for filename, reason in failed_files:
                                        st.error(f"❌ {filename}: {reason}")
                                
                                if added_count > 0 or failed_files:
                                    st.rerun()

            with tab3:
                st.markdown("<h3>🗑️ 删除图像</h3>", unsafe_allow_html=True)
                
                if "dataset_page" not in st.session_state:
                    st.session_state.dataset_page = 1
                
                images, total = get_all_images(st.session_state.dataset_page, per_page=20)
                
                if total == 0:
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                        border-radius: 15px;
                        padding: 3rem;
                        text-align: center;
                    ">
                        <h3 style="margin: 0 0 1rem 0; color: #2d3748; font-size: 1.5rem;">
                            📭 数据集为空
                        </h3>
                        <p style="margin: 0; color: #4a5568; font-size: 1.1rem;">
                            请先添加一些图像到数据集
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
                        border-radius: 12px;
                        padding: 1rem;
                        margin-bottom: 1.5rem;
                    ">
                        <p style="margin: 0; color: #4a5568;">
                            📊 共 <strong>{total}</strong> 张图片，当前第 <strong>{st.session_state.dataset_page}</strong> 页
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    MAX_COLS = 5
                    for row_start in range(0, len(images), MAX_COLS):
                        row_items = images[row_start : row_start + MAX_COLS]
                        cols = st.columns(len(row_items), gap="medium")
                        
                        for col, img_path in zip(cols, row_items):
                            with col:
                                try:
                                    st.image(
                                        str(img_path),
                                        caption=f"📁 {img_path.name}",
                                        use_container_width=True,
                                    )
                                except Exception as e:
                                    st.markdown(f"""
                                    <div style="
                                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
                                        border-radius: 10px;
                                        padding: 1rem;
                                        text-align: center;
                                    ">
                                        <p style="margin: 0; color: #4a5568;">
                                            {img_path.name}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                if st.button(
                                    f"🗑️ 删除",
                                    key=f"del_{img_path.name}",
                                    use_container_width=True
                                ):
                                    success, message = remove_image_from_dataset(img_path.name)
                                    if success:
                                        st.success(f"✅ {message}")
                                    else:
                                        st.error(f"❌ {message}")
                                    st.rerun()
                    
                    total_pages = (total + 19) // 20
                    if total_pages > 1:
                        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        with col1:
                            if st.button("⬅️ 上一页", disabled=st.session_state.dataset_page <= 1):
                                st.session_state.dataset_page -= 1
                                st.rerun()
                        
                        with col5:
                            if st.button("下一页 ➡️", disabled=st.session_state.dataset_page >= total_pages):
                                st.session_state.dataset_page += 1
                                st.rerun()
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🔄 重建索引", type="secondary", use_container_width=True):
                            with st.spinner("正在重建索引..."):
                                count = rebuild_index()
                                st.success(f"✅ 索引重建完成！共 {count} 张图片")
                                st.rerun()
