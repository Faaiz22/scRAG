"""
scRAG/app.py - Enhanced Streamlit app with multi-functional analysis platform
"""
import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Setup paths
proj_root = os.path.abspath(os.path.dirname(__file__))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.embeddings import Embedder
from src.vectorstore_faiss import FaissStore
from src.rag_chain import RAGChain
from src.ingestion import ensure_papers_folder, load_documents_from_papers_folder

# Optional scanpy import
try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    st.warning("⚠️ Scanpy not installed. Biological data analysis features will be limited.")

# Page config
st.set_page_config(page_title="scRAG Platform", layout="wide", page_icon="🧬")

# Initialize session state
if 'chain' not in st.session_state:
    st.session_state['chain'] = None
if 'index_loaded' not in st.session_state:
    st.session_state['index_loaded'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'adata' not in st.session_state:
    st.session_state['adata'] = None
if 'trained_classifier' not in st.session_state:
    st.session_state['trained_classifier'] = None
if 'trained_autoencoder' not in st.session_state:
    st.session_state['trained_autoencoder'] = None
if 'trained_generator' not in st.session_state:
    st.session_state['trained_generator'] = None
if 'trained_discriminator' not in st.session_state:
    st.session_state['trained_discriminator'] = None

# Title
st.title("🧬 scRAG – Comprehensive In Silico Analysis Platform")
st.markdown("**AI-powered platform for single-cell RNA-seq analysis and modeling**")

# Sidebar
st.sidebar.header("📚 Data Management")

# =========================
# BIOLOGICAL DATA UPLOAD
# =========================
if SCANPY_AVAILABLE:
    st.sidebar.subheader("🧬 Biological Data")
    uploaded_h5ad = st.sidebar.file_uploader(
        "Upload AnnData (.h5ad)",
        type=['h5ad'],
        help="Upload single-cell RNA-seq data in AnnData format"
    )
    
    if uploaded_h5ad is not None:
        if st.sidebar.button("📥 Load AnnData"):
            with st.spinner("Loading AnnData..."):
                try:
                    adata = sc.read_h5ad(uploaded_h5ad)
                    st.session_state['adata'] = adata
                    st.sidebar.success(f"✅ Loaded AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading AnnData: {str(e)}")
    
    # Show current AnnData info
    if st.session_state['adata'] is not None:
        adata = st.session_state['adata']
        st.sidebar.info(f"📊 Current data: {adata.n_obs} cells × {adata.n_vars} genes")
        
        # Preprocessing section
        with st.sidebar.expander("⚙️ Preprocessing"):
            st.caption("Run standard preprocessing steps")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Normalize", use_container_width=True):
                    with st.spinner("Normalizing..."):
                        sc.pp.normalize_total(adata, target_sum=1e4)
                        st.success("✅ Normalized")
                
                if st.button("Log Transform", use_container_width=True):
                    with st.spinner("Log transforming..."):
                        sc.pp.log1p(adata)
                        st.success("✅ Log transformed")
                
                if st.button("Find HVGs", use_container_width=True):
                    with st.spinner("Finding highly variable genes..."):
                        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
                        n_hvg = adata.var['highly_variable'].sum()
                        st.success(f"✅ Found {n_hvg} HVGs")
            
            with col2:
                if st.button("Run PCA", use_container_width=True):
                    with st.spinner("Running PCA..."):
                        sc.pp.pca(adata, n_comps=50)
                        st.success("✅ PCA done")
                
                if st.button("Compute Neighbors", use_container_width=True):
                    with st.spinner("Computing neighbors..."):
                        sc.pp.neighbors(adata)
                        st.success("✅ Neighbors computed")
                
                if st.button("Run UMAP", use_container_width=True):
                    with st.spinner("Running UMAP..."):
                        sc.tl.umap(adata)
                        st.success("✅ UMAP done")

st.sidebar.markdown("---")

# Papers directory
papers_dir = os.path.join("scRAG", "data", "papers")
ensure_papers_folder()

# Show papers in directory
st.sidebar.subheader("📄 Papers in Repository")
if os.path.exists(papers_dir):
    papers = sorted([p for p in os.listdir(papers_dir) if p.lower().endswith(('.pdf', '.docx', '.txt', '.md'))])
    if papers:
        for p in papers[:5]:
            st.sidebar.write(f"📄 {p}")
        if len(papers) > 5:
            st.sidebar.caption(f"... and {len(papers) - 5} more")
        st.sidebar.info(f"Total: {len(papers)} document(s)")
    else:
        st.sidebar.warning("No papers found.")

st.sidebar.markdown("---")

# File uploader for papers
st.sidebar.subheader("📤 Upload Papers")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, TXT, or MD files",
    type=['pdf', 'docx', 'txt', 'md'],
    accept_multiple_files=True
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
            st.error("❌ No documents found.")
        else:
            with st.spinner("Building index..."):
                try:
                    chain_tmp = RAGChain()
                    n = chain_tmp.rebuild_index_from_papers()
                    if n == 0:
                        st.error("❌ No documents processed.")
                    else:
                        st.success(f"✅ Index built: {n} chunks!")
                        st.session_state['index_loaded'] = False
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with col2:
    if st.button("📥 Load Index", use_container_width=True):
        with st.spinner("Loading index..."):
            try:
                embedder = Embedder()
                faiss_store = FaissStore(dim=embedder.embedding_dim())
                loaded = faiss_store.load()
                if not loaded:
                    st.warning("⚠️ No index found.")
                else:
                    chain = RAGChain(embedder=embedder, faiss_store=faiss_store)
                    st.session_state['chain'] = chain
                    st.session_state['index_loaded'] = True
                    st.success("✅ Index loaded!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if st.session_state['index_loaded']:
    st.sidebar.success("✅ Index ready")
else:
    st.sidebar.info("ℹ️ Index not loaded")

st.sidebar.markdown("---")
st.sidebar.caption("scRAG Platform v2.0")

# =========================
# MAIN CONTENT TABS
# =========================

tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 scRAG Chat",
    "🔬 Supervised Analysis",
    "🌌 Generative Models",
    "🤖 AI Co-pilot"
])

# =========================
# TAB 1: scRAG Chat
# =========================
with tab1:
    st.header("💬 Chat with Scientific Literature")
    
    if not st.session_state['index_loaded']:
        st.info("👈 Build and load the index from the sidebar to start chatting.")
    else:
        # Settings
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_area(
                "Ask a question about single-cell RNA-seq:",
                placeholder="e.g., What are the steps for quality control in scRNA-seq data?",
                height=100
            )
        with col2:
            k_value = st.slider("Top-k", 1, 20, 5)
            show_sources = st.checkbox("Show sources", True)
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit = st.button("🔍 Get Answer", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state['chat_history'] = []
                st.rerun()
        
        if submit and question.strip():
            with st.spinner("Thinking..."):
                try:
                    chain = st.session_state['chain']
                    answer, hits, dists = chain.answer(question, k=k_value)
                    st.session_state['chat_history'].insert(0, {
                        'question': question,
                        'answer': answer,
                        'hits': hits,
                        'dists': dists
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Display history
        if st.session_state['chat_history']:
            st.markdown("---")
            for idx, entry in enumerate(st.session_state['chat_history']):
                st.markdown(f"### 🙋 Question {len(st.session_state['chat_history']) - idx}")
                st.info(entry['question'])
                st.markdown("### 🤖 Answer")
                st.markdown(entry['answer'])
                
                if show_sources and entry['hits']:
                    with st.expander(f"📚 View {len([h for h in entry['hits'] if h])} Sources"):
                        for i, (hit, dist) in enumerate(zip(entry['hits'], entry['dists'])):
                            if hit:
                                st.markdown(f"**{i+1}. {hit.get('source', 'Unknown')}** (distance: {dist:.4f})")
                                st.caption(hit.get('excerpt', '')[:300])
                                st.markdown("---")
                st.markdown("---")

# =========================
# TAB 2: Supervised Analysis
# =========================
with tab2:
    st.header("🔬 Supervised Deep Learning Analysis")
    
    if not SCANPY_AVAILABLE:
        st.error("❌ Scanpy is required for this feature. Install with: pip install scanpy anndata")
    elif st.session_state['adata'] is None:
        st.info("👈 Upload an AnnData file from the sidebar to begin.")
    else:
        adata = st.session_state['adata']
        
        st.subheader("📊 Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cells", adata.n_obs)
        col2.metric("Genes", adata.n_vars)
        col3.metric("Observations", len(adata.obs.columns))
        
        st.markdown("---")
        st.subheader("🎯 Cell Type Classification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select label column
            label_cols = [col for col in adata.obs.columns if adata.obs[col].dtype in ['object', 'category']]
            if label_cols:
                label_col = st.selectbox("Select label column:", label_cols)
            else:
                st.warning("No categorical columns found in adata.obs")
                label_col = None
        
        with col2:
            # Select data representation
            data_options = []
            if 'X_pca' in adata.obsm:
                data_options.append('X_pca')
            if 'X_umap' in adata.obsm:
                data_options.append('X_umap')
            data_options.append('X (raw)')
            
            data_repr = st.selectbox("Data representation:", data_options)
        
        if st.button("🚀 Train Classifier", type="primary"):
            if label_col is None:
                st.error("❌ Please select a valid label column")
            else:
                with st.spinner("Training classifier..."):
                    try:
                        from src.models.supervised import SimpleClassifier
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import accuracy_score, classification_report
                        
                        # Extract data
                        if data_repr == 'X_pca':
                            X = adata.obsm['X_pca']
                        elif data_repr == 'X_umap':
                            X = adata.obsm['X_umap']
                        else:
                            X = adata.X
                            if hasattr(X, 'toarray'):
                                X = X.toarray()
                        
                        # Get labels
                        y = adata.obs[label_col].values
                        label_encoder = {label: i for i, label in enumerate(np.unique(y))}
                        y_encoded = np.array([label_encoder[label] for label in y])
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                        )
                        
                        # Create model
                        input_dim = X.shape[1]
                        n_classes = len(label_encoder)
                        model = SimpleClassifier(input_dim=input_dim, n_classes=n_classes)
                        
                        # Train
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model = model.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        criterion = torch.nn.CrossEntropyLoss()
                        
                        # Training loop
                        model.train()
                        epochs = 50
                        batch_size = 128
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for epoch in range(epochs):
                            indices = np.random.permutation(len(X_train))
                            for i in range(0, len(X_train), batch_size):
                                batch_idx = indices[i:i+batch_size]
                                X_batch = torch.FloatTensor(X_train[batch_idx]).to(device)
                                y_batch = torch.LongTensor(y_train[batch_idx]).to(device)
                                
                                optimizer.zero_grad()
                                outputs = model(X_batch)
                                loss = criterion(outputs, y_batch)
                                loss.backward()
                                optimizer.step()
                            
                            progress_bar.progress((epoch + 1) / epochs)
                            status_text.text(f"Epoch {epoch+1}/{epochs}")
                        
                        # Evaluate
                        model.eval()
                        with torch.no_grad():
                            X_test_tensor = torch.FloatTensor(X_test).to(device)
                            predictions = model(X_test_tensor).cpu().numpy()
                            y_pred = np.argmax(predictions, axis=1)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Store model
                        st.session_state['trained_classifier'] = model
                        
                        st.success(f"✅ Training complete! Accuracy: {accuracy:.2%}")
                        
                        # Get predictions for all data
                        model.eval()
                        with torch.no_grad():
                            X_all_tensor = torch.FloatTensor(X).to(device)
                            all_predictions = model(X_all_tensor).cpu().numpy()
                            y_pred_all = np.argmax(all_predictions, axis=1)
                        
                        # Reverse label encoding
                        reverse_encoder = {v: k for k, v in label_encoder.items()}
                        pred_labels = [reverse_encoder[i] for i in y_pred_all]
                        adata.obs['predicted_labels'] = pred_labels
                        
                        # Visualization
                        if 'X_umap' in adata.obsm:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                            
                            # True labels
                            sc.pl.umap(adata, color=label_col, ax=ax1, show=False, title='True Labels')
                            
                            # Predicted labels
                            sc.pl.umap(adata, color='predicted_labels', ax=ax2, show=False, title='Predicted Labels')
                            
                            st.pyplot(fig)
                        
                        # Classification report
                        st.subheader("📊 Classification Report")
                        report = classification_report(y_test, y_pred, target_names=[reverse_encoder[i] for i in sorted(reverse_encoder.keys())])
                        st.text(report)
                        
                    except Exception as e:
                        st.error(f"❌ Error training classifier: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

# =========================
# TAB 3: Generative Models
# =========================
with tab3:
    st.header("🌌 Generative Adversarial Networks")
    
    if not SCANPY_AVAILABLE:
        st.error("❌ Scanpy is required for this feature.")
    elif st.session_state['adata'] is None:
        st.info("👈 Upload an AnnData file from the sidebar to begin.")
    else:
        adata = st.session_state['adata']
        
        st.subheader("🔄 Step 1: Train Autoencoder")
        st.markdown("Compress high-dimensional data into latent space")
        
        col1, col2, col3 = st.columns(3)
        latent_dim = col1.number_input("Latent dimension", 32, 128, 64)
        ae_epochs = col2.number_input("Epochs", 10, 200, 50)
        ae_lr = col3.number_input("Learning rate", 0.0001, 0.01, 0.001, format="%.4f")
        
        if st.button("🔨 Train Autoencoder", type="primary"):
            with st.spinner("Training autoencoder..."):
                try:
                    from src.models.autoencoder import Autoencoder, train_autoencoder
                    
                    # Get data
                    if 'X_pca' in adata.obsm:
                        X = adata.obsm['X_pca']
                        st.info("Using PCA representation")
                    else:
                        X = adata.X
                        if hasattr(X, 'toarray'):
                            X = X.toarray()
                        st.info("Using raw expression data")
                    
                    input_dim = X.shape[1]
                    
                    # Create and train autoencoder
                    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    losses = train_autoencoder(
                        autoencoder, X, epochs=ae_epochs, lr=ae_lr, device=device
                    )
                    
                    st.session_state['trained_autoencoder'] = autoencoder
                    st.success("✅ Autoencoder trained successfully!")
                    
                    # Plot training loss
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(losses)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Reconstruction Loss')
                    ax.set_title('Autoencoder Training Loss')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error training autoencoder: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.markdown("---")
        st.subheader("🎭 Step 2: Train GAN on Latent Space")
        st.markdown("Generate synthetic cells in latent space")
        
        if st.session_state['trained_autoencoder'] is None:
            st.warning("⚠️ Train the autoencoder first!")
        else:
            col1, col2, col3, col4 = st.columns(4)
            z_dim = col1.number_input("Noise dimension", 16, 64, 32)
            gan_epochs = col2.number_input("GAN epochs", 50, 500, 100)
            gan_lr = col3.number_input("GAN learning rate", 0.0001, 0.01, 0.0002, format="%.4f")
            gan_batch = col4.number_input("Batch size", 32, 256, 128)
            
            if st.button("🎨 Train GAN", type="primary"):
                with st.spinner("Training GAN..."):
                    try:
                        from src.models.gan_latent import Generator, Discriminator
                        from src.models.autoencoder import get_latent_representations
                        
                        autoencoder = st.session_state['trained_autoencoder']
                        
                        # Get data
                        if 'X_pca' in adata.obsm:
                            X = adata.obsm['X_pca']
                        else:
                            X = adata.X
                            if hasattr(X, 'toarray'):
                                X = X.toarray()
                        
                        # Get latent representations
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        real_latents = get_latent_representations(autoencoder, X, device=device)
                        
                        # Create GAN
                        latent_dim = real_latents.shape[1]
                        generator = Generator(z_dim=z_dim, latent_dim=latent_dim).to(device)
                        discriminator = Discriminator(latent_dim=latent_dim).to(device)
                        
                        # Optimizers
                        opt_G = torch.optim.Adam(generator.parameters(), lr=gan_lr, betas=(0.5, 0.999))
                        opt_D = torch.optim.Adam(discriminator.parameters(), lr=gan_lr, betas=(0.5, 0.999))
                        criterion = torch.nn.BCEWithLogitsLoss()
                        
                        # Training loop
                        generator.train()
                        discriminator.train()
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        d_losses = []
                        g_losses = []
                        
                        for epoch in range(gan_epochs):
                            # Shuffle data
                            indices = np.random.permutation(len(real_latents))
                            
                            epoch_d_loss = 0
                            epoch_g_loss = 0
                            n_batches = 0
                            
                            for i in range(0, len(real_latents), gan_batch):
                                batch_idx = indices[i:i+gan_batch]
                                real_batch = torch.FloatTensor(real_latents[batch_idx]).to(device)
                                batch_size = real_batch.size(0)
                                
                                # Train Discriminator
                                opt_D.zero_grad()
                                
                                # Real samples
                                real_labels = torch.ones(batch_size, 1).to(device)
                                real_out = discriminator(real_batch)
                                d_loss_real = criterion(real_out, real_labels)
                                
                                # Fake samples
                                z = torch.randn(batch_size, z_dim).to(device)
                                fake_latents = generator(z)
                                fake_labels = torch.zeros(batch_size, 1).to(device)
                                fake_out = discriminator(fake_latents.detach())
                                d_loss_fake = criterion(fake_out, fake_labels)
                                
                                d_loss = d_loss_real + d_loss_fake
                                d_loss.backward()
                                opt_D.step()
                                
                                # Train Generator
                                opt_G.zero_grad()
                                z = torch.randn(batch_size, z_dim).to(device)
                                fake_latents = generator(z)
                                fake_out = discriminator(fake_latents)
                                g_loss = criterion(fake_out, real_labels)
                                g_loss.backward()
                                opt_G.step()
                                
                                epoch_d_loss += d_loss.item()
                                epoch_g_loss += g_loss.item()
                                n_batches += 1
                            
                            avg_d_loss = epoch_d_loss / n_batches
                            avg_g_loss = epoch_g_loss / n_batches
                            d_losses.append(avg_d_loss)
                            g_losses.append(avg_g_loss)
                            
                            progress_bar.progress((epoch + 1) / gan_epochs)
                            status_text.text(f"Epoch {epoch+1}/{gan_epochs} - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
                        
                        st.session_state['trained_generator'] = generator
                        st.session_state['trained_discriminator'] = discriminator
                        
                        st.success("✅ GAN trained successfully!")
                        
                        # Plot training curves
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(d_losses, label='Discriminator Loss', alpha=0.7)
                        ax.plot(g_losses, label='Generator Loss', alpha=0.7)
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.set_title('GAN Training Losses')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"❌ Error training GAN: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        st.markdown("---")
        st.subheader("🧪 Step 3: Generate Synthetic Cells")
        
        if st.session_state['trained_generator'] is None or st.session_state['trained_autoencoder'] is None:
            st.warning("⚠️ Train both the autoencoder and GAN first!")
        else:
            n_synthetic = st.number_input("Number of synthetic cells to generate", 100, 10000, 1000)
            
            if st.button("✨ Generate Synthetic Cells", type="primary"):
                with st.spinner("Generating synthetic cells..."):
                    try:
                        from src.models.autoencoder import reconstruct_from_latent
                        
                        generator = st.session_state['trained_generator']
                        autoencoder = st.session_state['trained_autoencoder']
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        
                        generator.eval()
                        autoencoder.eval()
                        
                        # Generate fake latent vectors
                        with torch.no_grad():
                            z_dim = generator.net[0].in_features
                            z = torch.randn(n_synthetic, z_dim).to(device)
                            fake_latents = generator(z).cpu().numpy()
                        
                        # Decode to high-dimensional space
                        synthetic_cells = reconstruct_from_latent(
                            autoencoder, fake_latents, device=device
                        )
                        
                        # Create new AnnData object with synthetic cells
                        synthetic_adata = ad.AnnData(X=synthetic_cells)
                        synthetic_adata.obs['cell_type'] = 'synthetic'
                        
                        # Combine with real data
                        real_adata = adata.copy()
                        real_adata.obs['cell_type'] = 'real'
                        
                        # Make sure dimensions match
                        if 'X_pca' in adata.obsm:
                            # Synthetic cells are already in PCA space
                            combined_X = np.vstack([real_adata.obsm['X_pca'], synthetic_cells])
                            combined_adata = ad.AnnData(X=combined_X)
                        else:
                            combined_adata = ad.concat([real_adata, synthetic_adata])
                        
                        combined_adata.obs['cell_type'] = (
                            ['real'] * real_adata.n_obs + ['synthetic'] * n_synthetic
                        )
                        
                        # Run UMAP on combined data
                        sc.pp.neighbors(combined_adata)
                        sc.tl.umap(combined_adata)
                        
                        st.success(f"✅ Generated {n_synthetic} synthetic cells!")
                        
                        # Visualize
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sc.pl.umap(combined_adata, color='cell_type', ax=ax, show=False,
                                   palette={'real': '#1f77b4', 'synthetic': '#ff7f0e'},
                                   title=f'Real vs Synthetic Cells (n={n_synthetic})')
                        st.pyplot(fig)
                        
                        # Statistics
                        col1, col2 = st.columns(2)
                        col1.metric("Real cells", real_adata.n_obs)
                        col2.metric("Synthetic cells", n_synthetic)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating synthetic cells: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

# =========================
# TAB 4: AI Co-pilot
# =========================
with tab4:
    st.header("🤖 AI-Powered Bioinformatics Co-pilot")
    st.markdown("Get executable Python code for your analysis tasks, powered by scientific literature")
    
    if not st.session_state['index_loaded']:
        st.info("👈 Build and load the index from the sidebar to use the AI Co-pilot.")
    else:
        st.markdown("""
        **How it works:**
        1. Ask a question about scRNA-seq analysis
        2. The AI retrieves relevant methods from scientific papers
        3. It generates executable `scanpy` code based on the literature
        4. You can copy and run the code directly!
        
        **Example questions:**
        - "How do I find marker genes for my clusters?"
        - "What's the best way to perform trajectory inference?"
        - "How can I integrate multiple datasets?"
        - "Show me how to do differential expression analysis"
        """)
        
        st.markdown("---")
        
        # Check if AnnData is loaded
        if st.session_state['adata'] is not None:
            adata = st.session_state['adata']
            st.success(f"✅ AnnData loaded: {adata.n_obs} cells × {adata.n_vars} genes")
            
            # Show available data
            with st.expander("📊 Available data in AnnData"):
                st.write("**Observations (adata.obs):**", list(adata.obs.columns))
                st.write("**Variables (adata.var):**", list(adata.var.columns))
                st.write("**Obsm (adata.obsm):**", list(adata.obsm.keys()))
        else:
            st.info("💡 Upload an AnnData file to get context-aware code generation")
        
        st.markdown("---")
        
        # Query interface
        copilot_question = st.text_area(
            "Ask the Co-pilot:",
            placeholder="e.g., How do I find marker genes for my clusters using the Leiden clustering?",
            height=100
        )
        
        k_copilot = st.slider("Number of papers to reference", 1, 10, 5, key="copilot_k")
        
        if st.button("🚀 Generate Code", type="primary"):
            if copilot_question.strip():
                with st.spinner("Consulting scientific literature and generating code..."):
                    try:
                        chain = st.session_state['chain']
                        
                        # Get relevant contexts
                        qvec = chain.embedder.embed_texts([copilot_question])[0]
                        hits, dists = chain.faiss.search(qvec, k=k_copilot)
                        
                        # Generate code using the co-pilot LLM
                        code_answer = chain.llm.generate(copilot_question, hits)
                        
                        st.markdown("### 📝 Generated Analysis Code")
                        st.code(code_answer, language="python")
                        
                        st.markdown("### 📚 Referenced Papers")
                        for i, (hit, dist) in enumerate(zip(hits, dists)):
                            if hit:
                                with st.expander(f"[{i+1}] {hit.get('source', 'Unknown')} (similarity: {1-dist:.3f})"):
                                    st.caption(hit.get('excerpt', '')[:500])
                        
                    except Exception as e:
                        st.error(f"❌ Error generating code: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question")
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Tips for better results:
        - Be specific about your analysis goal
        - Mention the clustering algorithm or method you're using
        - Reference specific columns in your data if applicable
        - Ask about standard workflows (QC, normalization, clustering, etc.)
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • PyTorch • Scanpy")
