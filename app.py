import os
import re
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Intelligence Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Ultra-Modern CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Overall Dark Gradient Canvas */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #090d16 100%) !important;
        color: #f1f5f9 !important;
    }

    /* Sidebar Custom Styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px) !important;
        padding-top: 1.5rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
    }

    /* Modern Title Branding */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.25rem;
    }
    
    .brand-title {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: -0.02em;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #4ade80;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 8px #22c55e;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    .brand-sub {
        color: #94a3b8 !important;
        font-size: 0.95rem;
        margin-bottom: 1.6rem;
    }

    /* Glassmorphic Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-bottom: 1.25rem;
    }

    .stat-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 14px;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(129, 140, 248, 0.4);
    }

    .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
        font-weight: 600;
    }

    .stat-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* Sleek Primary Button */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.4rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.55) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    /* Index Success Notification Box */
    .success-badge {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #6ee7b7;
        padding: 10px 14px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* High Contrast Chat Input */
    [data-testid="stChatInput"] {
        background-color: #1e293b !important;
        border: 1.5px solid #6366f1 !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45) !important;
        padding: 4px !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #ffffff !important;
        font-size: 1.02rem !important;
        caret-color: #818cf8 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
        font-weight: 400 !important;
    }

    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border-radius: 10px !important;
    }

    /* Chat Messages Styling */
    [data-testid="stChatMessage"] {
        background: rgba(30, 41, 59, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 1.15rem 1.35rem !important;
        margin-bottom: 1.1rem !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.2) 0%, rgba(30, 41, 59, 0.8) 100%) !important;
        border: 1px solid rgba(129, 140, 248, 0.35) !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        color: #f8fafc !important;
        font-size: 1.02rem !important;
        line-height: 1.7 !important;
    }

    [data-testid="stChatMessage"] strong {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        color: #c7d2fe !important;
        font-weight: 700 !important;
        margin-top: 0.9rem !important;
        margin-bottom: 0.4rem !important;
    }

    [data-testid="stChatMessage"] code {
        background: #0f172a !important;
        color: #38bdf8 !important;
        padding: 2px 7px !important;
        border-radius: 5px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* Quick Prompt Pills */
    .quick-chip {
        display: inline-flex;
        align-items: center;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #c7d2fe;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.84rem;
        cursor: pointer;
        margin-right: 8px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }

    /* Expandable Source Boxes */
    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 10px !important;
        margin-top: 0.8rem !important;
    }

    [data-testid="stExpander"] summary span {
        color: #a5b4fc !important;
        font-weight: 600 !important;
    }

    /* Sliders and Inputs */
    .stSlider [data-baseweb="slider"] {
        margin-top: 8px;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# Top Bar Header
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("""
    <div class="brand-container">
        <span style="font-size: 2rem;">⚡</span>
        <span class="brand-title">RAG Intelligence Studio</span>
        <div class="status-badge"><div class="status-dot"></div>Mistral 8B Active</div>
    </div>
    <div class="brand-sub">Next-generation multi-source RAG with Semantic ChromaDB retrieval & high-fidelity LLM synthesis</div>
    """, unsafe_allow_html=True)

with col_head2:
    st.write("")
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatMistralAI(model="ministral-8b-latest", temperature=0.2)

def sanitize_url(raw_url: str) -> str:
    raw_url = raw_url.strip()
    matches = list(re.finditer(r'https?://', raw_url))
    if len(matches) > 1:
        raw_url = raw_url[matches[-1].start():]
    return raw_url

def is_youtube_url(url: str) -> bool:
    return bool(re.search(r'(?:youtube\.com\/(?:watch\?v=|embed\/|shorts\/)|youtu\.be\/)', url))

def fetch_youtube_transcript(url: str):
    from youtube_transcript_api import YouTubeTranscriptApi
    m = re.search(r'(?:v=|youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})', url)
    if not m:
        raise ValueError("Invalid YouTube URL: could not detect 11-character video ID.")
    video_id = m.group(1)
    
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.list(video_id)
    try:
        t = transcript_list.find_transcript(['en', 'en-US', 'hi', 'es', 'fr', 'de'])
    except Exception:
        t = next(iter(transcript_list))
    fetched = t.fetch()
    text = " ".join([snippet.text for snippet in fetched.snippets])
    return [Document(
        page_content=text,
        metadata={"source": url, "video_id": video_id, "language": getattr(t, 'language', 'unknown')}
    )]

# Sidebar setup
with st.sidebar:
    st.markdown("### 🎛️ Knowledge Setup")
    
    doc_dir = Path("documents loader")
    available_files = []
    if doc_dir.exists():
        available_files = sorted([f.name for f in doc_dir.iterdir() if f.suffix in [".txt", ".pdf"] and not f.name.startswith("temp_")])
    
    def get_file_label(name: str) -> str:
        p = doc_dir / name
        if p.exists():
            kb = p.stat().st_size / 1024
            if kb >= 1024:
                return f"📑 {name} ({kb / 1024:.1f} MB ⚠️ Large)"
            return f"📄 {name} ({kb:.1f} KB)"
        return name

    source_choice = st.radio(
        "Source Category:",
        options=["Pre-loaded Documents", "Upload New File", "Web URL / YouTube Video"],
        index=0
    )
    
    loaded_docs = []
    doc_source_name = ""
    
    if source_choice == "Pre-loaded Documents":
        file_options = ["-- Select a file --"] + available_files
        default_idx = file_options.index("GRU.pdf") if "GRU.pdf" in file_options else (file_options.index("notes.txt") if "notes.txt" in file_options else 0)
        selected_file = st.selectbox(
            "Select Document:",
            options=file_options,
            index=default_idx,
            format_func=lambda x: get_file_label(x) if x in available_files else x
        )
        if selected_file and selected_file != "-- Select a file --":
            doc_source_name = selected_file
            file_path = doc_dir / selected_file
            try:
                if file_path.suffix == ".txt":
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    loaded_docs = loader.load()
                elif file_path.suffix == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                    loaded_docs = loader.load()
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                
    elif source_choice == "Upload New File":
        uploaded_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
        if uploaded_file:
            doc_source_name = uploaded_file.name
            temp_path = Path("documents loader") / f"temp_{uploaded_file.name}"
            temp_path.write_bytes(uploaded_file.getvalue())
            try:
                if uploaded_file.name.endswith(".txt"):
                    loader = TextLoader(str(temp_path), encoding="utf-8")
                    loaded_docs = loader.load()
                else:
                    loader = PyPDFLoader(str(temp_path))
                    loaded_docs = loader.load()
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                
    elif source_choice == "Web URL / YouTube Video":
        st.markdown("**Webpage or YouTube Video:**")
        input_url = st.text_input(
            "URL:",
            value="",
            placeholder="e.g. https://youtu.be/... or https://..."
        )
        
        if st.button("🌐 Ingest URL"):
            cleaned_url = sanitize_url(input_url)
            if not cleaned_url:
                st.warning("Please paste a valid URL.")
            else:
                with st.spinner("Extracting content..."):
                    try:
                        if is_youtube_url(cleaned_url):
                            docs = fetch_youtube_transcript(cleaned_url)
                            st.session_state["fetched_docs"] = docs
                            st.session_state["fetched_source_name"] = cleaned_url
                            st.session_state["is_youtube"] = True
                            st.session_state["yt_url"] = cleaned_url
                            st.success("YouTube transcript fetched!")
                        else:
                            loader = WebBaseLoader(cleaned_url)
                            docs = loader.load()
                            st.session_state["fetched_docs"] = docs
                            st.session_state["fetched_source_name"] = cleaned_url
                            st.session_state["is_youtube"] = False
                            st.success("Webpage fetched!")
                    except Exception as e:
                        st.error(f"Error fetching URL: {e}")
                        
        if "fetched_docs" in st.session_state:
            loaded_docs = st.session_state["fetched_docs"]
            doc_source_name = st.session_state.get("fetched_source_name", "Web/YouTube URL")

    st.markdown("---")
    st.markdown("### ⚙️ Chunking & Retrieval")
    chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=800, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=300, value=100, step=20)
    k_retrievals = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=6, value=3)

    st.markdown("---")
    # Explicit Indexing Button with Glowing Accent
    index_button = st.button("⚡ Index Document into ChromaDB")

    if index_button:
        if loaded_docs:
            with st.spinner(f"Chunking & embedding '{doc_source_name}'..."):
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = splitter.split_documents(loaded_docs)
                embeddings = get_embeddings()
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )
                st.session_state.current_indexed_source = doc_source_name
                st.session_state.chunk_count = len(chunks)
                st.session_state.index_success = True
        else:
            st.sidebar.warning("Select or fetch a document first.")

    if st.session_state.get("index_success") and st.session_state.get("vectorstore") is not None:
        st.markdown(f"""
        <div class="success-badge">
            <span>✨</span> <b>Indexed {st.session_state.get('chunk_count', 0)} Chunks Ready!</b>
        </div>
        """, unsafe_allow_html=True)

# Main Two-Column Layout
col1, col2 = st.columns([1.35, 1], gap="large")

with col1:
    st.markdown("### 💬 Interactive Assistant")
    
    # Quick starter prompt suggestions
    if st.session_state.get("vectorstore") is not None:
        st.caption("Quick actions:")
        qp1, qp2, qp3 = st.columns(3)
        quick_query = None
        with qp1:
            if st.button("📝 Summarize key ideas"):
                quick_query = "Please provide a concise summary of the key ideas and core findings."
        with qp2:
            if st.button("📐 Architecture details"):
                quick_query = "What is the architecture, technical methodology, or mathematical foundation described?"
        with qp3:
            if st.button("❓ Key Takeaways"):
                quick_query = "What are the most important takeaways and practical implications?"
    else:
        st.info("💡 **Getting Started:** Select a document on the left sidebar and click **'⚡ Index Document into ChromaDB'** to activate the assistant.")
        quick_query = None

    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"🔍 Retrieved Context Sources ({len(msg['sources'])} chunks)"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk #{i}:**")
                        st.caption(src.page_content)
                        st.markdown("---")

    chat_input_val = st.chat_input("Ask any question regarding the indexed knowledge base...")
    active_query = quick_query or chat_input_val

    if active_query:
        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)
            
        if st.session_state.get("vectorstore") is None:
            st.warning("⚠️ No document is currently indexed. Please select a document on the sidebar and click **'⚡ Index Document into ChromaDB'** first.")
            st.stop()
                
        with st.chat_message("assistant"):
            with st.spinner("Scanning vector embeddings..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k_retrievals})
                retrieved_docs = retriever.invoke(active_query)
                
                context_str = "\n\n---\n\n".join([d.page_content for d in retrieved_docs])
                
                prompt_template = ChatPromptTemplate([
                    ("system", "You are an intelligent, expert AI research assistant. Provide well-structured, clear, and comprehensive answers strictly grounded in the provided context. Use bullet points and bold headers where appropriate. If the context does not contain the answer, explicitly state what is missing."),
                    ("human", "Context:\n{context}\n\nUser Question: {question}\n\nStructured Answer:"),
                ])
                
                llm = get_llm()
                prompt = prompt_template.format_messages(context=context_str, question=active_query)

            # Stream tokens live (ChatGPT-style)
            def token_stream():
                for chunk in llm.stream(prompt):
                    if chunk.content:
                        yield chunk.content

            full_response = st.write_stream(token_stream())
            
            with st.expander(f"🔍 Retrieved Context Sources ({len(retrieved_docs)} chunks)"):
                for i, src in enumerate(retrieved_docs, 1):
                    st.markdown(f"**Chunk #{i}:**")
                    st.caption(src.page_content)
                    st.markdown("---")
                    
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": retrieved_docs
            })

with col2:
    st.markdown("### 📊 Knowledge Telemetry")
    
    current_source = st.session_state.get('current_indexed_source', doc_source_name or 'None')
    total_chunks = st.session_state.get('chunk_count', 0)
    
    st.markdown(f"""
    <div class="metric-grid">
        <div class="stat-card">
            <div class="stat-label">Active Source</div>
            <div class="stat-val" title="{current_source}">{"📑 " + current_source if current_source != "None" else "None"}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Indexed Chunks</div>
            <div class="stat-val">{total_chunks} Chunks</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Vector Embedding</div>
            <div class="stat-val" title="sentence-transformers/all-MiniLM-L6-v2">MiniLM-L6-v2 (384d)</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">LLM Model</div>
            <div class="stat-val">Ministral 8B</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.get("is_youtube") and "yt_url" in st.session_state:
        st.markdown("#### 📺 Video Player")
        st.video(st.session_state["yt_url"])
        
    st.markdown("#### 📄 Document Preview")
    if loaded_docs:
        preview_text = loaded_docs[0].page_content[:1200]
        st.text_area(
            label="Live Raw Excerpt",
            value=preview_text + ("\n... [Truncated for preview]" if len(loaded_docs[0].page_content) > 1200 else ""),
            height=260,
            disabled=True
        )
    else:
        st.info("Select a document or load a URL to preview its text content here.")
