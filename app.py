"""
scRAG/app.py - Streamlit app to interact with the scRAG index and LLM.
"""
import streamlit as st
import os, sys
proj_root = os.path.abspath(os.path.dirname(__file__))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.embeddings import Embedder
from src.vectorstore_faiss import FaissStore
from src.rag_chain import RAGChain

st.set_page_config(page_title="scRAG", layout="wide")
st.title("scRAG — Single-cell RAG Assistant (Demo)")

papers_dir = os.path.join("scRAG", "data", "papers")
st.sidebar.markdown("### Papers in repo")
if os.path.exists(papers_dir):
    for p in sorted(os.listdir(papers_dir)):
        st.sidebar.write("- " + p)
else:
    st.sidebar.write("No papers folder found.")

if st.sidebar.button("Rebuild index from scRAG/data/papers"):
    with st.spinner("Rebuilding index (this may take a few minutes)"):
        chain_tmp = RAGChain()
        n = chain_tmp.rebuild_index_from_papers()
        if n == 0:
            st.error("No documents found in scRAG/data/papers/")
        else:
            st.success(f"Rebuilt index with {n} chunks. Saved to scRAG/data/")

if st.sidebar.button("Load index"):
    with st.spinner("Loading index..."):
        embedder = Embedder()
        faiss_store = FaissStore(dim=embedder.embedding_dim())
        loaded = faiss_store.load()
        if not loaded:
            st.warning("No index found. Rebuild the index first.")
        else:
            st.success("Index loaded.")
            chain = RAGChain(embedder=embedder, faiss_store=faiss_store)
            st.session_state['chain'] = chain

if 'chain' not in st.session_state:
    st.info("Load or rebuild index from the sidebar to start querying.")

q = st.text_input("Ask about scRNA-seq analysis, models or methods")
k = st.slider("Top-k retrieval", 1, 10, 5)
if st.button("Get answer"):
    if 'chain' not in st.session_state:
        st.error("Index not loaded. Load or rebuild index first.")
    else:
        chain = st.session_state['chain']
        with st.spinner("Retrieving and generating..."):
            answer, hits, dists = chain.answer(q, k=k)
        st.subheader("Answer")
        st.code(answer)
        st.subheader("Top retrieved chunks")
        for i, (h,d) in enumerate(zip(hits,dists)):
            if not h: continue
            st.markdown(f"**{i+1}. {h.get('source')}** — distance: `{d:.3f}`")
            st.caption(h.get('source_path'))
            st.write(h.get('excerpt'))
