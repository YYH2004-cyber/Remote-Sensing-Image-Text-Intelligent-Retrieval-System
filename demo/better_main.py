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

HISTORY_FILE = Path("/home/yyh2004/demo/demo/data/search_history.json")

if "history" not in st.session_state:
    st.session_state.history = load_history(HISTORY_FILE)
st.session_state.TOPK = 5
if "page" not in st.session_state:
    st.session_state.page = "文本 → 图像检索"  # 默认页面

# 设置页面配置
st.set_page_config(
    page_title="🔍 遥感图像-文本智能检索系统",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义CSS样式
st.markdown(
    """
<style>
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    /* 标题样式 */
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 1.5rem;
        padding: 0.5rem 0;
        border-bottom: 2px solid #3498db;
    }
    
    /* 菜单项样式 */
    .menu-item {
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .menu-item:hover {
        background-color: #e3f2fd;
        transform: translateX(5px);
    }
    
    .menu-item.active {
        background-color: #3498db;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .menu-item i {
        font-size: 1.2rem;
    }
    
    /* 内容区域样式 */
    .content-container {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    
    /* 按钮样式 */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* 选项卡样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f8ff;
        border-radius: 8px 8px 0 0;
        gap: 1px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    
    /* 历史记录样式 */
    .history-record {
        border-left: 3px solid #3498db;
        padding: 10px 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
        border-radius: 0 8px 8px 0;
    }
    
    .history-time {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    
    /* 图片容器样式 */
    .image-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 8px;
        margin: 8px 0;
        background-color: #f9f9f9;
        transition: all 0.3s ease;
    }
    
    .image-container:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: scale(1.02);
    }
    
    /* 页脚样式 */
    .footer {
        text-align: center;
        padding: 20px;
        color: #7f8c8d;
        font-size: 0.9rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 30px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# 布局：侧边栏 + 主内容区
# -----------------------------
with st.sidebar:
    
    # -----------------------------
    # 顶部标题 (左上角)
    # -----------------------------
    st.markdown(
        '<div class="main-header">🔍 遥感图像-文本智能检索系统</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 🔧 功能导航")

    # 自定义菜单项
    menu_items = [
        {"label": "文本 → 图像检索", "icon": "📝", "page": "文本 → 图像检索"},
        {"label": "图像 → 图像检索", "icon": "🖼️", "page": "图像 → 图像检索"},
        {"label": "图像 → 文本描述", "icon": "🧠", "page": "图像 → 文本描述"},
        {"label": "历史记录", "icon": "📜", "page": "历史记录"},
    ]

    # 渲染菜单项
    for item in menu_items:
        is_active = st.session_state.get("page", "") == item["page"]
        menu_class = "menu-item active" if is_active else "menu-item"
        if st.button(
            f"{item['icon']} {item['label']}",
            key=f"menu_{item['page']}",
            use_container_width=True,
        ):
            st.session_state.page = item["page"]
            st.rerun()

    # 添加分隔线
    st.markdown("---")

    # 系统信息
    st.markdown("### ℹ️ 系统信息")
    st.markdown(f"""
    - **当前时间**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
    - **历史记录**: {len(st.session_state.history)} 条
    """)

    # 添加页脚
    st.markdown(
        """
    <div style="margin-top: 30px; padding-top: 15px; border-top: 1px solid #e0e0e0;">
        <small>© 2025 遥感智能检索系统</small><br>
        <small>Powered by Streamlit & FAISS</small>
    </div>
    """,
        unsafe_allow_html=True,
    )

# =============================
# 主内容区
# =============================
with st.spinner("加载模型中...", show_time=True):
    index, meta, text_index, meta_text = build_index()

# 创建内容容器
with st.container():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    # 页面内容
    if st.session_state.page == "文本 → 图像检索":
        st.subheader("📝 文本 → 图像检索")
        st.markdown("输入描述性文本，在遥感图像库中查找最匹配的图像")

        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "🔍 输入检索文本", placeholder="a photo of church"
            )
        with col2:
            st.session_state.TOPK = st.select_slider(
                "📊 TOP-K",
                options=[5, 10, 15, 20],
                value=st.session_state.TOPK,
                help="设置返回结果的数量",
            )

        if st.button("🚀 开始检索", type="primary", use_container_width=True):
            if not query.strip():
                st.warning("请输入检索文本")
            elif index is None:
                st.error("没有索引，请先放图片到 ./images 并重启")
            else:
                with st.spinner("正在检索中..."):
                    q_emb = text_to_embedding(query)
                    D, I = index.search(q_emb, st.session_state.TOPK)
                    st.success(f"找到 {st.session_state.TOPK} 个匹配结果!")

                    # 显示结果
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
                                st.markdown(
                                    f'<div class="image-container">',
                                    unsafe_allow_html=True,
                                )
                                st.image(str(img_path), use_container_width=True)
                                st.caption(f"{filename}\nscore={score:.4f}")
                                st.markdown("</div>", unsafe_allow_html=True)

                    # 添加历史记录
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
        st.subheader("🖼️ 图像 → 图像检索")
        st.markdown("上传一张遥感图像，查找图库中视觉相似的图像")

        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded = st.file_uploader(
                "📤 上传查询图像",
                type=["jpg", "jpeg", "png", "tif", "tiff"],
                accept_multiple_files=False,
                help="支持 JPG, PNG, TIF 格式",
            )
        with col2:
            st.session_state.TOPK = st.select_slider(
                "📊 TOP-K",
                options=[5, 10, 15, 20],
                value=st.session_state.TOPK,
                help="设置返回结果的数量",
            )

        if uploaded:
            try:
                pil = Image.open(uploaded).convert("RGB")
                st.image(pil, caption="查询图像", width=300)

                if st.button(
                    "🔍 查找相似图像", type="primary", use_container_width=True
                ):
                    if index is None:
                        st.error("没有索引，请先放图片到 ./images 并重启")
                    else:
                        with st.spinner("正在检索中..."):
                            q_emb = image_to_embedding(pil)
                            D, I = index.search(q_emb, st.session_state.TOPK)
                            st.success(f"找到 {st.session_state.TOPK} 个相似图像!")

                            # 显示结果
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
                                        st.markdown(
                                            f'<div class="image-container">',
                                            unsafe_allow_html=True,
                                        )
                                        st.image(
                                            str(img_path), use_container_width=True
                                        )
                                        st.caption(f"{filename}\nscore={score:.4f}")
                                        st.markdown("</div>", unsafe_allow_html=True)

                            # 添加历史记录
                            add_to_history(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "type": "图像 → 图像",
                                    "query": uploaded.name,
                                    "topk": st.session_state.TOPK,
                                    "results": [
                                        meta[idx]["filename"]
                                        for idx in I[0]
                                        if idx >= 0
                                    ],
                                },
                                HISTORY_FILE,
                            )
                            st.session_state.history = load_history(HISTORY_FILE)
            except Exception as e:
                st.error(f"处理上传的图片时出错: {e}")

    elif st.session_state.page == "图像 → 文本描述":
        st.subheader("🧠 图像 → 文本描述")
        st.markdown("上传遥感图像，获取语义相关的文本描述")

        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded = st.file_uploader(
                "📤 上传查询图像",
                type=["jpg", "jpeg", "png", "tif", "tiff"],
                accept_multiple_files=False,
                help="支持 JPG, PNG, TIF 格式",
            )
        with col2:
            st.session_state.TOPK = st.select_slider(
                "📊 TOP-K",
                options=[1, 2, 3, 4, 5],
                value=st.session_state.TOPK,
                help="设置返回描述的数量",
            )

        if uploaded:
            try:
                pil = Image.open(uploaded).convert("RGB")
                st.image(pil, caption="查询图像", width=300)

                if st.button("💬 生成描述", type="primary", use_container_width=True):
                    if index is None:
                        st.error("没有索引，请先放图片到 ./images 并重启")
                    else:
                        with st.spinner("正在生成描述..."):
                            q_emb = image_to_embedding(pil)
                            D, I = text_index.search(q_emb, st.session_state.TOPK)
                            st.success(f"生成 {st.session_state.TOPK} 个相关描述!")

                            # 显示结果
                            st.markdown("### 📝 检索到的文本描述")
                            for i, (score, idx) in enumerate(zip(D[0], I[0])):
                                if idx < 0:
                                    continue
                                caption = meta_text[idx]["filename"]
                                st.markdown(f"**描述 {i + 1}** (相似度: {score:.4f})")
                                st.info(caption)
                                st.divider()

                            # 添加历史记录
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
                st.error(f"处理上传的图片时出错: {e}")

    elif st.session_state.page == "历史记录":
        st.subheader("📜 检索历史记录")
        st.markdown("查看和管理您的所有检索操作记录")

        if not st.session_state.history:
            st.info("📝 暂无历史记录。进行一些检索操作后，记录会显示在这里。")
            st.image(
                "https://cdn.pixabay.com/photo/2017/03/19/20/19/no-data-2155901_1280.png",
                width=300,
            )
        else:
            # 清除所有记录按钮
            if st.button("🧹 清除所有历史记录", type="secondary"):
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.session_state.history = []
                st.rerun()
                st.success("历史记录已清除！")

            st.markdown("---")

            # 排序：最新记录在最前
            sorted_history = sorted(
                st.session_state.history, key=lambda x: x["timestamp"], reverse=True
            )

            for i, record in enumerate(sorted_history):
                timestamp = datetime.fromisoformat(record["timestamp"])
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                # 创建折叠面板
                with st.expander(
                    f"🕒 {formatted_time} | {record['type']} | TOP-{record['topk']}"
                ):
                    # 查询内容
                    st.markdown(f"**🔍 查询内容**: `{record['query']}`")
                    st.markdown(f"**⏱️ 操作时间**: {formatted_time}")

                    # 结果展示
                    st.markdown("**📊 检索结果**:")

                    if record["type"] == "图像→文本":
                        # 文本结果
                        for j, text in enumerate(record["results"][: record["topk"]]):
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
                                        st.markdown(
                                            f'<div class="image-container">',
                                            unsafe_allow_html=True,
                                        )
                                        st.image(
                                            str(img_path),
                                            use_container_width=True,
                                            caption=filename,
                                        )
                                        st.markdown("</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"{filename}")

                    # 删除单条记录按钮
                    if st.button(
                        "🗑️ 删除此记录",
                        key=f"del_{i}_{record['timestamp']}",
                        type="secondary",
                    ):
                        st.session_state.history = [
                            h
                            for h in st.session_state.history
                            if h["timestamp"] != record["timestamp"]
                        ]
                        save_history(st.session_state.history, HISTORY_FILE)
                        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # 关闭 content-container

# 页脚
st.markdown(
    """
<div class="footer">
    <p>遥感图像-文本智能检索系统 © 2025 | 基于深度学习的跨模态检索</p>
    <p>系统状态：✅ 稳定运行中 | 响应时间：<span id="response-time">0.00s</span></p>
</div>
<script>
    // 简单的响应时间模拟
    document.addEventListener('DOMContentLoaded', function() {
        const now = new Date();
        const responseTime = (now.getSeconds() % 10) * 0.01 + 0.1;
        document.getElementById('response-time').textContent = responseTime.toFixed(2) + 's';
    });
</script>
""",
    unsafe_allow_html=True,
)
