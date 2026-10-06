import os
import tempfile
import streamlit as st
from ingestion import parse_file
from chunking import process_document_pages
from vector_store import VectorStore
import database
from llm import LocalRAGLLM

st.set_page_config(
    page_title="Campus Knowledge AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek Dark Mode Theme CSS
st.markdown("""
<style>
    /* Global Background & Font */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp {
        background-color: #0d1117 !important;
        color: #f0f6fc !important;
    }

    /* Sidebar Dark Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
        padding-top: 1.5rem;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Center Chat Area */
    .block-container {
        max-width: 820px !important;
        padding-top: 2rem !important;
        padding-bottom: 6rem !important;
    }

    /* Transparent Chat Bubbles */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0.85rem 0rem !important;
    }

    /* User Message Bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.25rem !important;
        margin-bottom: 0.75rem;
    }

    /* Assistant Response Area */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        padding: 0.5rem 0.25rem !important;
        color: #f0f6fc !important;
    }

    /* Dark Mode Source Expander */
    div[data-testid="stExpander"] {
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        background-color: #161b22 !important;
        margin-top: 0.6rem !important;
    }

    /* Minimal Citation Badges */
    .citation-badge {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 600;
        color: #60a5fa;
        background-color: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(96, 165, 250, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        margin-bottom: 4px;
    }

    /* Document item pill in sidebar */
    .doc-stat-pill {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0d1117;
        border: 1px solid #30363d;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 0.82rem;
    }

    /* Dark Chat Input Bar */
    div[data-testid="stChatInput"] {
        background-color: #0d1117 !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: #161b22 !important;
        color: #f0f6fc !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stChatInput"] textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }

    /* Action buttons */
    button[kind="secondary"] {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
        color: #f0f6fc !important;
    }
    button[kind="secondary"]:hover {
        border-color: #8b949e !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_services():
    store = VectorStore()
    llm = LocalRAGLLM()
    return store, llm

v_store, rag_llm = get_services()

# --- SIDEBAR: Workspace & Document Library ---
with st.sidebar:
    st.markdown("### 🎓 University AI")
    st.caption("Institutional Knowledge Assistant")

    st.markdown("---")
    st.markdown("##### 📁 Ingestion Center")
    
    uploaded_files = st.file_uploader(
        "Upload Syllabi or Guidelines",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("Index Selected Files", use_container_width=True, type="primary"):
            total_added = 0
            with st.spinner("Processing & embedding vectors..."):
                for uploaded_file in uploaded_files:
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        pages = parse_file(tmp_path)
                        for p in pages:
                            p["filename"] = uploaded_file.name
                        
                        chunks = process_document_pages(pages, chunk_size=1500, chunk_overlap=200)
                        added = v_store.add_documents(chunks)
                        total_added += added
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

            st.success(f"Indexed {total_added} chunks.")
            st.rerun()

    st.markdown("##### Knowledge Repository")
    stats = database.get_document_stats()
    if stats:
        for doc in stats:
            st.markdown(f"""
            <div class="doc-stat-pill">
                <span style="font-weight: 500; color: #e6edf3; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 170px;">
                    📄 {doc['filename']}
                </span>
                <span class="citation-badge">{doc['chunk_count']} chunks</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No files indexed yet.")

    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome Screen (Shown before conversation starts)
if not st.session_state.messages:
    st.markdown("### How can I help you today?")
    st.caption("Ask questions about prerequisites, credit policies, grading criteria, or institutional guidelines.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Graduation requirements", use_container_width=True):
            st.session_state.preset_query = "What are the requirements for graduation?"
    with col2:
        if st.button("📊 Credit transfer policy", use_container_width=True):
            st.session_state.preset_query = "What is the policy regarding course credit transfer?"

# Render Chat History
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "🧑‍💻" if role == "user" else "🎓"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"🔍 Verified Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"<span class='citation-badge'>{s['filename']} • Page {s['page_number']}</span>", unsafe_allow_html=True)
                    st.caption(f"> {s['text_content']}")

# User Query Handler
query = st.chat_input("Ask about regulations, syllabi, course policies...")
if "preset_query" in st.session_state and st.session_state.preset_query:
    query = st.session_state.preset_query
    st.session_state.preset_query = None

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(query)

    context_chunks = v_store.search(query, top_k=3)

    with st.chat_message("assistant", avatar="🎓"):
        if not context_chunks:
            fallback = "I could not find relevant institutional records. Please upload documents via the sidebar first."
            st.markdown(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback, "sources": []})
        else:
            stream_gen = rag_llm.stream_answer(query, context_chunks)
            full_response = st.write_stream(stream_gen)

            with st.expander(f"🔍 Verified Sources ({len(context_chunks)})"):
                for s in context_chunks:
                    st.markdown(f"<span class='citation-badge'>{s['filename']} • Page {s['page_number']}</span>", unsafe_allow_html=True)
                    st.caption(f"> {s['text_content']}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": context_chunks
            })