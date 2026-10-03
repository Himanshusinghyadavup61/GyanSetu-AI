# 🧠 GyanSetu-AI (RAG Intelligence Studio)

> **An advanced Retrieval-Augmented Generation (RAG) platform with multi-source ingestion (PDFs, Webpages, YouTube Videos), semantic ChromaDB search, and real-time streaming LLM synthesis.**

[![Author](https://img.shields.io/badge/Author-Himanshu%20Singh%20Yadav-blue?style=for-the-badge&logo=github)](https://github.com/Himanshusinghyadavup61)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C?style=for-the-badge)](https://langchain.com/)

---

## 👨‍💻 Created by
**Himanshu Singh Yadav**

---

## 🚀 Key Features

- **Multi-Source Document Ingestion**:
  - 📄 **PDFs**: Full text parsing and chunking (`pypdf`).
  - 📝 **Plain Text**: Supports notes, documentation, and raw transcripts.
  - 🌐 **Web Pages**: Live webpage scraping and content extraction (`WebBaseLoader`).
  - 📺 **YouTube Videos**: Native video transcript extraction across languages (including auto-generated Hindi & English captions via `youtube-transcript-api`).
- **Semantic Vector Storage**:
  - Powered by **ChromaDB** with sentence-level semantic representations.
  - Embeddings generated via `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **ChatGPT-Style Streaming**:
  - Token-by-token streaming via `st.write_stream` and `llm.stream()` for sub-500ms time-to-first-token.
- **Glassmorphic UI**:
  - Interactive telemetry metrics, document preview, YouTube player, and source citation inspector.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Data Sources: PDF / TXT / Web / YouTube] --> B[LangChain Document Loaders]
    B --> C[Recursive Character Text Splitter]
    C --> D[HuggingFace Embeddings: all-MiniLM-L6-v2]
    D --> E[(ChromaDB Vector Store)]
    
    F[User Query] --> G[Similarity Search / Retriever]
    E --> G
    G --> H[Top-K Relevant Chunks]
    
    H --> I[Prompt Template + Context]
    F --> I
    I --> J[Mistral AI: ministral-8b-latest]
    J --> K[Real-time Token Stream to Streamlit UI]
```

---

## 🛠️ Tech Stack

- **Framework**: LangChain (`langchain`, `langchain-community`, `langchain-chroma`, `langchain-mistralai`)
- **LLM**: Mistral AI (`ministral-8b-latest`)
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Frontend**: Streamlit
- **Package Manager**: `uv`

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Himanshusinghyadavup61/GyanSetu-AI.git
cd GyanSetu-AI
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your actual API keys:
```env
MISTRAL_API_KEY="your-mistral-api-key"
OPENAI_API_KEY="your-openai-api-key"
GROQ_API_KEY="your-groq-api-key"
```

### 3. Install Dependencies
Using [`uv`](https://github.com/astral-sh/uv):
```bash
uv sync
```

---

## 🖥️ Running the Application

### Launch Interactive Web App
```bash
uv run streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### Run CLI Pipelines
- **Test Main RAG Pipeline**:
  ```bash
  uv run python main.py
  ```
- **Test Vector Store Search**:
  ```bash
  uv run python "vector store/db.py"
  ```
- **Test PDF Loader**:
  ```bash
  uv run python "documents loader/pdf.py"
  ```

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
