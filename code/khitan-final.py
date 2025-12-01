import streamlit as st
import pandas as pd
import os
import re
import json
import uuid
import unicodedata
from openai import OpenAI
from datetime import datetime

# ================= 1. 页面配置 =================
st.set_page_config(
    page_title="契丹小字溯源解析系统",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 2. 增强版 CSS (UI 核心修复) =================
# ================= 2. 深色模式 CSS (替换原 CSS) =================
st.markdown("""
<style>
    /* 全局深色背景 */
    .stApp {
        background-color: #0e1117; /* 深灰黑 */
        color: #fafafa;
    }

    /* 标题样式 */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #e0e0e0; /* 浅灰字 */
        text-align: center;
        margin-top: -20px;
        margin-bottom: 5px;
        letter-spacing: 2px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #9b59b6; /* 紫色微调 */
        text-align: center;
        margin-bottom: 2.5rem;
        font-family: "Courier New", monospace;
        font-weight: bold;
    }

    /* --- 输入区域优化 --- */
    div.stButton > button:first-child {
        height: 46px;
        width: 100%;
        border: 1px solid #4a4a4a;
        background-color: #262730;
        color: white;
    }
    
    /* 输入框深色适配 */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 6px;
        border: 1px solid #4a4a4a;
        background-color: #262730;
        color: white;
        height: 46px;
    }

    /* --- 侧边栏 --- */
    section[data-testid="stSidebar"] {
        background-color: #262730; /* 侧边栏深色 */
        border-right: 1px solid #4a4a4a;
    }
    
    /* 侧边栏按钮 */
    div[data-testid="stSidebar"] .stButton button {
        background-color: #1e1e1e;
        color: #ecf0f1;
        border: 1px solid #4a4a4a;
        border-left: 4px solid #57606f;
        text-align: left;
        padding: 10px;
    }
    div[data-testid="stSidebar"] .stButton button:hover {
        border-left: 4px solid #e74c3c;
        background-color: #2d3436;
        color: #ffffff;
    }
    
    /* --- 结果卡片深色化 --- */
    .result-container {
        background-color: #1e1e1e; /* 卡片深色 */
        border: 1px solid #4a4a4a;
        border-top: 4px solid #3498db;
        border-radius: 8px;
        padding: 30px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.3);
        margin-top: 20px;
        position: relative;
    }
    .result-meta {
        position: absolute;
        top: 15px;
        right: 20px;
        font-size: 0.85rem;
        color: #7f8c8d;
    }
    
    /* 针对 Markdown 输出的文字颜色修正 */
    .result-container h3 {
        color: #ecf0f1 !important;
    }
    .result-container p, .result-container li {
        color: #bdc3c7 !important;
    }
</style>
""", unsafe_allow_html=True)

# ================= 3. 全局配置与数据 (功能保持不变) =================
CSV_FILE = "契丹小字_清洗后训练集.csv"
TXT_FILE = "khitan_phonetic.txt"
HISTORY_FILE = "khitan_history_v13.json"

class HistoryManager:
    @staticmethod
    def load_history():
        if not os.path.exists(HISTORY_FILE): return []
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return []

    @staticmethod
    def save_record(query, mode, result, context_count):
        history = HistoryManager.load_history()
        new_record = {
            "id": str(uuid.uuid4()),
            "query": query,
            "mode": mode,
            "result": result,
            "context_count": context_count,
            "timestamp": datetime.now().strftime("%m-%d %H:%M")
        }
        history.insert(0, new_record)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history[:30], f, ensure_ascii=False, indent=2)
        return new_record

    @staticmethod
    def clear_history():
        if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)

    @staticmethod
    def delete_record(record_id):
        history = HistoryManager.load_history()
        history = [h for h in history if h['id'] != record_id]
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

class KhitanLogic:
    @staticmethod
    def normalize_pinyin(s):
        if not s: return ""
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn').lower()

    @staticmethod
    @st.cache_data
    def load_data(csv_path, txt_path):
        v_data_list = []
        debug_info = [] 
        
        if os.path.exists(csv_path):
            try:
                encodings = ['utf-8', 'utf-8-sig', 'gbk']
                df = None
                for enc in encodings:
                    try:
                        temp_df = pd.read_csv(csv_path, encoding=enc, on_bad_lines='skip')
                        if len(temp_df.columns) > 1: df = temp_df; break
                    except: continue
                if df is not None:
                    df.columns = [str(c).lower().strip() for c in df.columns]
                    src_col = next((c for c in df.columns if any(x in c for x in ['source', 'word', '契丹'])), df.columns[0])
                    tgt_col = next((c for c in df.columns if any(x in c for x in ['target', 'meaning', '中文'])), df.columns[1])
                    for _, row in df.iterrows():
                        w, m = str(row[src_col]).strip(), str(row[tgt_col]).strip()
                        if w and m and m.lower() != 'nan':
                            v_data_list.append({'word': w, 'meaning': m, 'pronunciation': '', 'pinyin_norm': '', 'type': 'csv'})
                    debug_info.append(f"✅ 字形库: {len(df)} 条")
            except Exception as e: debug_info.append(f"❌ CSV 错误: {e}")

        if os.path.exists(txt_path):
            pattern = re.compile(r"^\s*([^\(\s]+)\s*(?:[\(（](.+?)[\)）])?\s*[:：]\s*(.+)$")
            count = 0
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or len(line) < 2 or " - " in line: continue 
                        match = pattern.match(line)
                        if match:
                            v_data_list.append({'word': match.group(1).strip(), 'meaning': match.group(3).strip(), 'pronunciation': match.group(2).strip() if match.group(2) else "", 'pinyin_norm': KhitanLogic.normalize_pinyin(match.group(2).strip() if match.group(2) else ""), 'type': 'txt'})
                            count += 1
                        elif '：' in line or ':' in line:
                            sep = '：' if '：' in line else ':'
                            parts = line.split(sep, 1)
                            v_data_list.append({'word': parts[0].strip(), 'meaning': parts[1].strip(), 'pronunciation': '', 'pinyin_norm': '', 'type': 'txt'})
                            count += 1
                debug_info.append(f"✅ 拟音库: {count} 条")
            except Exception as e: debug_info.append(f"❌ TXT 错误: {e}")
        return v_data_list, debug_info

    @staticmethod
    def get_smart_context(v_data_list, query, mode="c2k"):
        context_items = []
        found_keywords = set()
        query = query.strip()
        query_norm = KhitanLogic.normalize_pinyin(query)
        tokens = query.split()
        if len(tokens) == 1 and len(tokens[0]) > 1: tokens.extend(list(tokens[0]))
        matched_entries = []
        for item in v_data_list:
            w, m, p, p_norm = item['word'], item['meaning'], item['pronunciation'], item['pinyin_norm']
            is_match, match_type, match_token = False, "", ""
            for token in tokens:
                if not token.strip(): continue
                if token == w: is_match, match_type, match_token = True, "原字精确", token
                elif token in m: is_match, match_type, match_token = True, "含义包含", token
                elif m in token: is_match, match_type, match_token = True, "含义相关", token
            if not is_match and item['type'] == 'txt' and len(query_norm) > 1 and p_norm:
                if query_norm == p_norm: is_match, match_type, match_token = True, "拼音精确", query
                elif query_norm in p_norm: is_match, match_type, match_token = True, "拼音模糊", query
            if is_match:
                if not any(x['word'] == w and x['meaning'] == m for x in matched_entries):
                    item_copy = item.copy()
                    item_copy['match_token'] = match_token
                    item_copy['match_type'] = match_type
                    matched_entries.append(item_copy)
                    if match_token: found_keywords.add(match_token)
        matched_entries.sort(key=lambda x: (0 if x['match_type'] == '拼音精确' else 1, 0 if x['match_type'] == '原字精确' else 1, -len(x['word'])))
        top_matches = matched_entries[:50]
        for item in top_matches:
            if item['type'] == 'csv':
                step_tag = "第一步：契丹小字(字形)"
                content = f"字形码[{item['word']}] 原义：{item['meaning']}"
            else:
                step_tag = "第二步：契丹汉字(拟音)"
                pron_display = f"({item['pronunciation']})" if item['pronunciation'] else ""
                content = f"写法：{item['word']}{pron_display} -> 古义：{item['meaning']}"
            context_items.append(f"【{step_tag}|{item['match_type']}】{content}")
        full_text = "\n".join(context_items)
        if not full_text: full_text = "（本地资料库无直接匹配，请基于语言学知识推理）"
        return full_text, len(context_items), list(found_keywords)

# ================= 4. UI 初始化 =================
if 'active_record' not in st.session_state:
    st.session_state.active_record = None

v_data_list, debug_msg = KhitanLogic.load_data(CSV_FILE, TXT_FILE)

# ================= 5. 侧边栏 (视觉优化) =================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scroll.png", width=50)
    st.markdown("### ⚙️ 设置")
    api_key = st.text_input("sk-1002144e1e1747cf9a257c68ce98776d", type="password", help="在此输入 Key")
    
    with st.expander("📊 数据库状态", expanded=False):
        for msg in debug_msg:
            if "✅" in msg: st.markdown(f"<span style='color:green; font-size:14px'>{msg}</span>", unsafe_allow_html=True)
            elif "❌" in msg: st.markdown(f"<span style='color:red; font-size:14px'>{msg}</span>", unsafe_allow_html=True)
            else: st.info(msg)

    st.divider()
    
    # 历史记录头部
    c1, c2 = st.columns([4, 1])
    c1.markdown("### 🕒 探索历史")
    if c2.button("🗑️", help="清空所有"): 
        HistoryManager.clear_history()
        st.session_state.active_record = None
        st.rerun()

    history = HistoryManager.load_history()
    # 历史记录列表 - 样式优化版
    with st.container(height=500):
        if not history:
            st.caption("暂无查询记录...")
        for rec in history:
            with st.container():
                # 使用 Columns 布局，左边大按钮，右边小删除
                col_main, col_del = st.columns([5, 1])
                
                # 左侧：查询内容按钮
                icon = "🔤" if '汉' in rec['mode'] else "📜"
                # 截断过长文本
                disp_query = (rec['query'][:10] + '..') if len(rec['query']) > 10 else rec['query']
                
                if col_main.button(f"{icon} {disp_query}", key=rec['id'], help=f"{rec['timestamp']} | {rec['query']}", use_container_width=True):
                    st.session_state.active_record = rec
                    st.rerun()
                
                # 右侧：删除按钮 (利用CSS类 small-btn)
                st.markdown('<div class="small-btn">', unsafe_allow_html=True)
                if col_del.button("✖", key=f"del_{rec['id']}"):
                    HistoryManager.delete_record(rec['id'])
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 4px'></div>", unsafe_allow_html=True)

# ================= 6. 主界面 (布局修复) =================

st.markdown('<div class="main-title">📜 Khitan Small Script Origin Analysis System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Modern Vernacular ⇌ Khitan Script Transliteration & Glyphs</div>', unsafe_allow_html=True)

# 输入控制台 - 使用 Container 包裹，限制宽度防止在大屏上太散
with st.container():
    # 使用 vertical_alignment="bottom" (Streamlit 1.36+ 支持)
    # 如果版本较低，上面的 CSS 'margin-top' 已经处理了对齐
    c_mode, c_input, c_btn = st.columns([1.5, 3.5, 1]) 
    
    with c_mode:
        mode = st.selectbox(
            "🏳️ 溯源模式", 
            ["汉 -> 契丹 (翻译/造词)", "契丹 -> 汉 (溯源/解析)"],
            index=1
        )
        mode_code = "c2k" if "汉 ->" in mode else "k2c"

    with c_input:
        ph = "输入现代汉语 (如: 皇帝)..." if mode_code == "c2k" else "输入契丹字、汉字拟音或拼音 (如: linya)..."
        query = st.text_input("✏️ 查询内容", placeholder=ph, label_visibility="visible")

    with c_btn:
        # 添加一个空label来占位，或者依赖CSS对齐
        st.write("") 
        st.write("") 
        # Primary 按钮颜色较鲜艳
        start_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

    # 逻辑处理
    if start_btn:
        if not api_key: st.error("⚠️ 请在左侧侧边栏设置 API Key")
        elif not query: st.warning("⚠️ 请输入查询内容")
        else:
            st.session_state.active_record = None
            
            # 进度条
            progress_bar = st.progress(0, text="正在启动神经语言学引擎...")
            
            try:
                # 1. 检索
                progress_bar.progress(30, text="正在检索本地史料库 (CSV/TXT)...")
                context_text, ctx_count, _ = KhitanLogic.get_smart_context(v_data_list, query, mode_code)
                
                # 2. 推理
                progress_bar.progress(60, text="DeepSeek 正在构建三步逻辑链...")
                
                if mode_code == "c2k":
                    system_prompt = f"你是一位契丹语言文字专家...【参考资料】\n{context_text}..." # (保持原 Prompt 内容，为节省长度省略)
                    system_prompt = f"""你是一位契丹语言文字专家。
【任务】将现代汉语转换为契丹小字逻辑链。
【逻辑链条】请严格遵循：现代白话 -> 契丹汉字(拟音) -> 契丹小字(字形)。
【参考资料】\n{context_text}
【回答要求】
1. **现代白话**：确认用户输入词汇的准确含义[可以适当使用语义相近词替换]。
2. **契丹汉字(拟音)**：查找对应的契丹语音译（如“阿保机”、“林牙”）。
3. **契丹小字**：如果资料中有对应的字形记录，请列出；否则基于拟音进行推测。
4. **输出格式**：输出完整契丹小字溯源链："""
                else:
                    system_prompt = f"""你是一位契丹语言文字专家。
【任务】对输入的契丹词汇进行“三步走”溯源解析。
【核心逻辑】契丹小字 -> 契丹汉字 -> 现代白话。
【参考资料】\n{context_text}
【执行步骤】请严格按以下格式输出（支持 Markdown）：
### 1. 🧬 第一步：契丹小字 (原始形态)
*   **字形状态**：(根据参考资料中的【第一步】数据...)
### 2. 🗣️ 第二步：契丹汉字 (拟音/借字)
*   **书写形式**：(引用参考资料【第二步】中的汉字...)
*   **发音标注**：(如有拼音请标注...)
### 3. 📝 第三步：现代白话 (通俗语义)
*   **古义今译**：(关键步骤，必须翻译为前三条相似的：各词含义拼接之后最接近的/现代大白话/一句古诗与出处...)
"""

                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"分析对象：{query}"}
                    ],
                    temperature=0.3 
                )
                res_text = response.choices[0].message.content
                new_rec = HistoryManager.save_record(query, mode_code, res_text, ctx_count)
                st.session_state.active_record = new_rec
                
                progress_bar.progress(100, text="✅ 分析完成")
                st.rerun()
                
            except Exception as e:
                st.error(f"发生错误: {e}")
                progress_bar.empty()

# ================= 7. 结果展示 (卡片化) =================
if st.session_state.active_record:
    rec = st.session_state.active_record
    
    # 使用 HTML/CSS 构建卡片容器
    st.markdown(f"""
    <div class="result-container">
        <div class="result-meta">📅 分析时间: {rec['timestamp']}</div>
        <h3 style="color:#2c3e50; border-bottom:2px solid #ecf0f1; padding-bottom:10px; margin-top:0;">
            💡 分析报告: <span style="color:#e67e22">{rec['query']}</span>
        </h3>
    """, unsafe_allow_html=True)
    
    # 结果正文
    st.markdown(rec['result'])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 证据链折叠区
    st.write("")
    with st.expander(f"📚 查看原始史料证据 (命中 {rec.get('context_count', 0)} 条记录)"):
        # 重新获取上下文用于展示
        ctx, count, _ = KhitanLogic.get_smart_context(v_data_list, rec['query'], rec['mode'])
        if count == 0:
            st.warning("本次分析基于 AI 纯逻辑推理，未在本地数据库中找到直接匹配项。")
        else:
            st.markdown(f"```text\n{ctx}\n```")
# ================= 8. 版权信息 ( "本地知识库 + 大模型推理" 架构：RAG (检索增强生成) )，下一步三个进步方向 =================
#1.进一步收集预料，对应契丹小字（长句子）平行语料库
#2.通过利用用户已经生成的正确内容作为参考，作为后续查询的上下文，提升准确率和一致性
#3.可以自己通过训练模型专门针对契丹小字进行微调，提升专业度