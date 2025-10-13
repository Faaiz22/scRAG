"""
scRAG/app.py - Streamlit app to interact with the scRAG index and LLM.
"""
import streamlit as st
import os
import sys

# Setup paths
proj_root = os.path.abspath(os.path.dirname(__file__))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.embeddings import Embedder
from src.vectorstore_faiss import FaissStore
from src.rag_chain import RAGChain
from src.ingestion import ensure_papers_folder, load_documents_from_papers_folder

# Page config
st.set_page_config(page_title="scRAG", layout="wide", page_icon="🧬")

# Initialize session state
if 'chain' not in st.session_state:
    st.session_state['chain'] = None
if 'index_loaded' not in st.session_state:
    st.session_state['index_loaded'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Title and description
st.title("🧬 scRAG – Single-cell RNA-seq RAG Assistant")
st.markdown("**Ask questions about single-cell RNA-seq analysis, models, and methods based on your uploaded papers.**")

# Sidebar
st.sidebar.header("📚 Document Management")

# Papers directory
papers_dir = os.path.join("scRAG", "data", "papers")
ensure_papers_folder()

# Show papers in directory
st.sidebar.subheader("Papers in Repository")
if os.path.exists(papers_dir):
    papers = sorted([p for p in os.listdir(papers_dir) if p.lower().endswith(('.pdf', '.docx', '.txt', '.md'))])
    if papers:
        for p in papers:
            st.sidebar.write(f"📄 {p}")
        st.sidebar.info(f"Total: {len(papers)} document(s)")
    else:
        st.sidebar.warning("No papers found. Please upload papers to `scRAG/data/papers/`")
else:
    st.sidebar.error("Papers folder not found!")

st.sidebar.markdown("---")

# File uploader
st.sidebar.subheader("📤 Upload Papers")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, TXT, or MD files",
    type=['pdf', 'docx', 'txt', 'md'],
    accept_multiple_files=True,
    help="Upload research papers or documents to add to the knowledge base"
)

if uploaded_files:
    if st.sidebar.button("💾 Save Uploaded Files"):
        saved_count = 0
        for uploaded_file in uploaded_files:
            file_path = os.path.join(papers_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_count += 1
        st.sidebar.success(f"✅ Saved {saved_count} file(s)!")
        st.rerun()

st.sidebar.markdown("---")

# Index management
st.sidebar.subheader("🔧 Index Management")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🔨 Build Index", use_container_width=True):
        docs = load_documents_from_papers_folder()
        if not docs:
            st.error("❌ No documents found in scRAG/data/papers/. Please upload papers first.")
        else:
            with st.spinner("🔄 Building index (this may take a few minutes)..."):
                try:
                    chain_tmp = RAGChain()
                    n = chain_tmp.rebuild_index_from_papers()
                    if n == 0:
                        st.error("❌ No documents could be processed.")
                    else:
                        st.success(f"✅ Index built successfully with {n} chunks!")
                        st.session_state['index_loaded'] = False
                        st.session_state['chain'] = None
                except Exception as e:
                    st.error(f"❌ Error building index: {str(e)}")

with col2:
    if st.button("📥 Load Index", use_container_width=True):
        with st.spinner("🔄 Loading index..."):
            try:
                embedder = Embedder()
                faiss_store = FaissStore(dim=embedder.embedding_dim())
                loaded = faiss_store.load()
                if not loaded:
                    st.warning("⚠️ No index found. Please build the index first.")
                else:
                    chain = RAGChain(embedder=embedder, faiss_store=faiss_store)
                    st.session_state['chain'] = chain
                    st.session_state['index_loaded'] = True
                    st.success("✅ Index loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading index: {str(e)}")

# Show index status
if st.session_state['index_loaded']:
    st.sidebar.success("✅ Index is loaded and ready")
else:
    st.sidebar.info("ℹ️ Index not loaded. Build or load index to start.")

st.sidebar.markdown("---")

# Settings
st.sidebar.subheader("⚙️ Settings")
k_value = st.sidebar.slider("Top-k retrieval", min_value=1, max_value=20, value=5, help="Number of relevant chunks to retrieve")
show_sources = st.sidebar.checkbox("Show source documents", value=True, help="Display retrieved document chunks")
show_distances = st.sidebar.checkbox("Show similarity scores", value=False, help="Display distance scores for retrieved chunks")

# Main content area
if not st.session_state['index_loaded']:
    st.info("👈 **Getting Started:** Upload papers and build the index using the sidebar controls.")
    
    with st.expander("📖 How to use this app"):
        st.markdown("""
        1. **Upload Papers**: Use the sidebar to upload PDF, DOCX, TXT, or MD files
        2. **Build Index**: Click "Build Index" to process and embed your documents
        3. **Load Index**: Click "Load Index" to load the processed index
        4. **Ask Questions**: Type your question in the text box and get answers!
        
        **Example Questions:**
        - What are the best practices for scRNA-seq normalization?
        - How do I perform clustering on single-cell data?
        - What models are recommended for cell type annotation?
        - Explain the steps for differential expression analysis
        """)
else:
    # Query interface
    st.markdown("### 💬 Ask a Question")
    
    # Query input
    question = st.text_area(
        "Enter your question about single-cell RNA-seq:",
        placeholder="e.g., What are the steps for quality control in scRNA-seq data?",
        height=100,
        help="Ask about methods, models, analysis pipelines, or best practices"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear History", use_container_width=True)
    
    if clear_button:
        st.session_state['chat_history'] = []
        st.rerun()
    
    # Process query
    if submit_button and question.strip():
        if st.session_state['chain'] is None:
            st.error("❌ Index not loaded. Please load the index first.")
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    chain = st.session_state['chain']
                    answer, hits, dists = chain.answer(question, k=k_value)
                    
                    # Add to chat history
                    st.session_state['chat_history'].insert(0, {
                        'question': question,
                        'answer': answer,
                        'hits': hits,
                        'dists': dists
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
    
    # Display chat history
    if st.session_state['chat_history']:
        st.markdown("---")
        
        for idx, entry in enumerate(st.session_state['chat_history']):
            with st.container():
                st.markdown(f"### 🙋 Question {len(st.session_state['chat_history']) - idx}")
                st.info(entry['question'])
                
                st.markdown("### 🤖 Answer")
                st.markdown(entry['answer'])
                
                if show_sources and entry['hits']:
                    with st.expander(f"📚 View {len([h for h in entry['hits'] if h])} Retrieved Sources", expanded=False):
                        for i, (hit, dist) in enumerate(zip(entry['hits'], entry['dists'])):
                            if not hit:
                                continue
                            
                            source_title = hit.get('source', 'Unknown')
                            source_path = hit.get('source_path', '')
                            excerpt = hit.get('text', hit.get('excerpt', ''))[:500]
                            
                            st.markdown(f"**{i+1}. {source_title}**")
                            if show_distances:
                                st.caption(f"Distance: {dist:.4f}")
                            st.caption(f"Path: {source_path}")
                            st.text_area(f"Excerpt {i+1}", excerpt, height=100, disabled=True, key=f"excerpt_{idx}_{i}")
                            st.markdown("---")
                
                st.markdown("---")
    elif st.session_state['index_loaded']:
        st.info("💡 **Tip:** Enter a question above to get started!")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("scRAG Demo v1.0")
st.sidebar.caption("Single-cell RNA-seq RAG Assistant")
