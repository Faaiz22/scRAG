"""
scRAG/src/rag_chain.py

Enhanced RAG orchestration with AI Co-pilot mode for generating executable code.
Supports optional local HF model via HF_MODEL env var; otherwise uses mock.
"""
import os
from typing import List, Tuple
from src.embeddings import Embedder
from src.vectorstore_faiss import FaissStore
from src.chunking import docs_to_chunks
from src.ingestion import load_documents_from_papers_folder

# Optional HF support
HF_AVAILABLE = False
try:
    from transformers import pipeline
    import torch
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

class LLMWrapper:
    def __init__(self, hf_env_key: str = "HF_MODEL"):
        self.hf_model_name = os.environ.get(hf_env_key, None)
        self.hf_device = os.environ.get("HF_DEVICE", None)
        self.max_tokens = int(os.environ.get("HF_MAX_TOKENS", 512))
        self.temperature = float(os.environ.get("HF_TEMPERATURE", 0.0))
        self.pipeline = None
        self.is_seq2seq = False
        
        if self.hf_model_name and HF_AVAILABLE:
            print(f"Attempting to load HF model: {self.hf_model_name}")
            try:
                device = 0 if (self.hf_device == "cuda" or (self.hf_device is None and torch.cuda.is_available())) else -1
                try:
                    self.pipeline = pipeline("text2text-generation", model=self.hf_model_name, device=device)
                    self.is_seq2seq = True
                    print(f"✓ Loaded seq2seq model on device {device}")
                except Exception:
                    self.pipeline = pipeline("text-generation", model=self.hf_model_name, device=device)
                    self.is_seq2seq = False
                    print(f"✓ Loaded text-generation model on device {device}")
            except Exception as e:
                print(f"✗ Failed to load HF model {self.hf_model_name}: {e}")
                print("Falling back to mock LLM")
                self.pipeline = None
        else:
            if not self.hf_model_name:
                print("No HF_MODEL environment variable set. Using mock LLM.")
            elif not HF_AVAILABLE:
                print("HuggingFace transformers not available. Using mock LLM.")

    def generate(self, question: str, contexts: List[dict], mode: str = "qa") -> str:
        """
        Generate answer or code based on question and retrieved contexts.
        
        Args:
            question: User's question
            contexts: List of retrieved document contexts
            mode: Generation mode - "qa" for Q&A, "copilot" for code generation
            
        Returns:
            Generated text (answer or code)
        """
        # Build context block from retrieved documents
        ctx_lines = []
        for i, c in enumerate(contexts[:8]):
            if not c:
                continue
            title = c.get("source", "unknown")
            excerpt = (c.get("excerpt") or c.get("text", "")[:400]).replace("\n", " ")
            ctx_lines.append(f"[{i+1}] {title} -- {excerpt[:400]}")
        
        context_block = "\n".join(ctx_lines) if ctx_lines else "No relevant context found."
        
        # Choose prompt based on mode
        if mode == "copilot":
            prompt = self._build_copilot_prompt(question, context_block)
        else:
            prompt = self._build_qa_prompt(question, context_block)
        
        # Generate using HF model if available
        if self.pipeline is not None:
            try:
                if self.is_seq2seq:
                    out = self.pipeline(
                        prompt,
                        max_length=self.max_tokens,
                        do_sample=(self.temperature > 0.0),
                        temperature=self.temperature if self.temperature > 0.0 else None
                    )
                    return out[0]["generated_text"].strip()
                else:
                    out = self.pipeline(
                        prompt,
                        max_new_tokens=self.max_tokens,
                        do_sample=(self.temperature > 0.0),
                        temperature=self.temperature if self.temperature > 0.0 else None
                    )
                    # Extract only the new generated text
                    generated = out[0]["generated_text"]
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()
                    return generated
            except Exception as e:
                print(f"✗ HF generation failed: {e}")
                print("Falling back to mock answer")
                if mode == "copilot":
                    return self._mock_copilot(question, contexts)
                else:
                    return self._mock_answer(question, contexts)
        
        # Use mock answer if no HF model
        if mode == "copilot":
            return self._mock_copilot(question, contexts)
        else:
            return self._mock_answer(question, contexts)

    def _build_qa_prompt(self, question: str, context_block: str) -> str:
        """Build prompt for Q&A mode."""
        prompt = (
            "You are a helpful assistant specializing in single-cell genomics and RNA-seq analysis.\n"
            "Use the retrieved evidence below to answer the user's question accurately and concisely.\n\n"
            "Retrieved evidence:\n"
            f"{context_block}\n\n"
            "Instructions:\n"
            "1. Answer based on the evidence provided\n"
            "2. Be specific and cite sources using [n] notation\n"
            "3. If asked about methods, provide step-by-step instructions\n"
            "4. For model recommendations, include architecture and hyperparameters\n"
            "5. If the evidence doesn't contain the answer, say so honestly\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return prompt

    def _build_copilot_prompt(self, question: str, context_block: str) -> str:
        """Build prompt for AI Co-pilot code generation mode."""
        prompt = (
            "You are an expert Bioinformatics Co-pilot specializing in single-cell RNA-seq analysis.\n"
            "The user has loaded an AnnData object named 'adata' containing single-cell RNA-seq data.\n\n"
            "TASK: Generate clean, executable Python code using scanpy to answer the user's question.\n\n"
            "Retrieved scientific evidence:\n"
            f"{context_block}\n\n"
            "INSTRUCTIONS:\n"
            "1. Generate ONLY executable Python code (no explanations outside comments)\n"
            "2. Use the 'adata' variable which contains the user's AnnData object\n"
            "3. Use scanpy (sc) functions based on the methods described in the retrieved papers\n"
            "4. Include inline comments citing the source using [n] notation\n"
            "5. Handle common scenarios:\n"
            "   - Clustering: Use sc.tl.leiden() or sc.tl.louvain()\n"
            "   - Marker genes: Use sc.tl.rank_genes_groups()\n"
            "   - Trajectory: Use sc.tl.paga() or sc.tl.dpt()\n"
            "   - Differential expression: Use sc.tl.rank_genes_groups() with appropriate method\n"
            "   - Visualization: Use sc.pl.* functions\n"
            "6. Assume standard preprocessing is done (normalization, PCA, neighbors)\n"
            "7. If preprocessing is needed, include those steps with comments\n"
            "8. Make code ready to copy-paste and execute\n\n"
            f"User Question: {question}\n\n"
            "Generated Python Code:\n"
            "```python\n"
        )
        return prompt

    def _mock_answer(self, question: str, contexts: List[dict]) -> str:
        """Generate an intelligent answer by extracting and synthesizing information from contexts."""
        if not contexts or all(not c for c in contexts):
            return self._fallback_answer(question)
        
        # Extract relevant text from all contexts
        all_text = []
        sources = []
        for i, c in enumerate(contexts[:5]):
            if not c:
                continue
            text = c.get('text', c.get('excerpt', ''))
            if text:
                all_text.append(text)
                sources.append((i+1, c.get('source', 'Unknown')))
        
        if not all_text:
            return self._fallback_answer(question)
        
        # Combine all context
        combined_text = ' '.join(all_text).lower()
        question_lower = question.lower()
        
        # Generate answer based on question type and context
        answer = self._extract_answer(question_lower, combined_text, sources, contexts)
        
        return answer
    
    def _extract_answer(self, question: str, context: str, sources: list, contexts: list) -> str:
        """Extract and format answer from context based on question."""
        lines = []
        
        # Determine question type and extract relevant info
        if any(kw in question for kw in ['what is', 'define', 'definition', 'explain']):
            # Definition/explanation question
            subject = self._extract_subject(question)
            lines.append(f"Based on the retrieved literature:\n")
            
            # Extract definition-like sentences
            definitions = self._extract_definitions(context, subject, contexts)
            if definitions:
                for def_text, src_idx in definitions[:3]:
                    lines.append(f"[{src_idx}] {def_text}\n")
            
            # Add key points
            key_points = self._extract_key_points(context, subject)
            if key_points:
                lines.append("\n**Key Points:**")
                for point in key_points[:5]:
                    lines.append(f"- {point}")
        
        elif any(kw in question for kw in ['how to', 'how do', 'steps', 'process', 'workflow']):
            # How-to/process question
            lines.append("Based on the retrieved literature:\n")
            
            # Extract step-by-step information
            steps = self._extract_steps(context, contexts)
            if steps:
                lines.append("\n**Approach:**")
                for i, (step, src_idx) in enumerate(steps[:6], 1):
                    lines.append(f"{i}. {step} [{src_idx}]")
            else:
                # Extract method descriptions
                methods = self._extract_methods(context, contexts)
                for method, src_idx in methods[:4]:
                    lines.append(f"- {method} [{src_idx}]")
        
        elif any(kw in question for kw in ['best', 'recommended', 'should', 'compare', 'better']):
            # Recommendation/comparison question
            lines.append("Based on the retrieved literature:\n")
            
            recommendations = self._extract_recommendations(context, contexts)
            if recommendations:
                for rec, src_idx in recommendations[:4]:
                    lines.append(f"- {rec} [{src_idx}]")
        
        elif any(kw in question for kw in ['why', 'reason', 'because', 'advantage', 'disadvantage']):
            # Reasoning/explanation question
            lines.append("Based on the retrieved literature:\n")
            
            explanations = self._extract_explanations(context, contexts)
            for exp, src_idx in explanations[:4]:
                lines.append(f"- {exp} [{src_idx}]")
        
        else:
            # General question - extract most relevant sentences
            lines.append("Based on the retrieved literature:\n")
            relevant = self._extract_relevant_sentences(question, contexts)
            for sent, src_idx in relevant[:5]:
                lines.append(f"[{src_idx}] {sent}\n")
        
        # Add sources
        if sources:
            lines.append("\n**Sources:**")
            for idx, source in sources:
                lines.append(f"[{idx}] {source}")
        
        return "\n".join(lines) if lines else self._fallback_answer(question)
    
    def _extract_definitions(self, context: str, subject: str, contexts: list) -> list:
        """Extract definition-like sentences."""
        definitions = []
        patterns = [
            ' is ', ' are ', ' refers to ', ' means ', ' defined as ',
            ' involves ', ' comprises ', ' consists of '
        ]
        
        for i, c in enumerate(contexts[:5]):
            if not c:
                continue
            text = c.get('text', '')
            sentences = text.split('. ')
            
            for sent in sentences:
                sent_lower = sent.lower()
                if any(pattern in sent_lower for pattern in patterns):
                    # Clean up the sentence
                    sent = sent.strip()
                    if len(sent) > 30 and len(sent) < 300:
                        definitions.append((sent + '.', i+1))
        
        return definitions[:3]
    
    def _extract_key_points(self, context: str, subject: str) -> list:
        """Extract key points about a subject."""
        key_phrases = []
        
        # Look for sentences with key indicators
        indicators = [
            'important', 'key', 'main', 'primary', 'essential',
            'significant', 'critical', 'crucial', 'breakthrough',
            'allows', 'enables', 'provides', 'offers'
        ]
        
        sentences = context.split('. ')
        for sent in sentences:
            if any(ind in sent.lower() for ind in indicators):
                sent = sent.strip()
                if 20 < len(sent) < 200:
                    # Capitalize first letter
                    sent = sent[0].upper() + sent[1:] if sent else sent
                    if not sent.endswith('.'):
                        sent += '.'
                    key_phrases.append(sent)
        
        return key_phrases[:5]
    
    def _extract_steps(self, context: str, contexts: list) -> list:
        """Extract step-by-step information."""
        steps = []
        
        # Look for numbered steps or sequential indicators
        step_indicators = [
            'first', 'second', 'third', 'then', 'next', 'finally',
            'step', 'stage', 'phase', 'followed by'
        ]
        
        for i, c in enumerate(contexts[:5]):
            if not c:
                continue
            text = c.get('text', '')
            sentences = text.split('. ')
            
            for sent in sentences:
                sent_lower = sent.lower()
                if any(ind in sent_lower for ind in step_indicators):
                    sent = sent.strip()
                    if 20 < len(sent) < 250:
                        sent = sent[0].upper() + sent[1:] if sent else sent
                        if not sent.endswith('.'):
                            sent += '.'
                        steps.append((sent, i+1))
        
        return steps[:6]
    
    def _extract_methods(self, context: str, contexts: list) -> list:
        """Extract method descriptions."""
        methods = []
        
        method_keywords = [
            'method', 'approach', 'technique', 'algorithm', 'tool',
            'using', 'apply', 'perform', 'employ', 'utilize'
        ]
        
        for i, c in enumerate(contexts[:5]):
            if not c:
                continue
            text = c.get('text', '')
            sentences = text.split('. ')
            
            for sent in sentences:
                if any(kw in sent.lower() for kw in method_keywords):
                    sent = sent.strip()
                    if 25 < len(sent) < 250:
                        sent = sent[0].upper() + sent[1:] if sent else sent
                        if not sent.endswith('.'):
                            sent += '.'
                        methods.append((sent, i+1))
        
        return methods[:5]
    
    def _extract_recommendations(self, context: str, contexts: list) -> list:
        """Extract recommendations."""
        recommendations = []
        
        rec_keywords = [
            'recommend', 'suggest', 'should', 'best', 'optimal',
            'preferred', 'advised', 'commonly used', 'widely used',
            'standard', 'suitable', 'appropriate'
        ]
        
        for i, c in enumerate(contexts[:5]):
            if not c:
                continue
            text = c.get('text', '')
            sentences = text.split('. ')
            
            for sent in sentences:
                if any(kw in sent.lower() for kw in rec_keywords):
                    sent = sent.strip()
                    if 25 < len(sent) < 250:
                        sent = sent[0].upper() + sent[1:] if sent else sent
                        if not sent.endswith('.'):
                            sent += '.'
                        recommendations.append((sent, i+1))
        
        return recommendations[:5]
    
    def _extract_explanations(self, context: str, contexts: list) -> list:
        """Extract explanatory sentences."""
        explanations = []
        
        exp_keywords = [
            'because', 'since', 'due to', 'reason', 'cause',
            'therefore', 'thus', 'consequently', 'as a result',
            'advantage', 'benefit', 'disadvantage', 'limitation'
        ]
        
        for i, c in enumerate(contexts[:5]):
            if not c:
                continue
            text = c.get('text', '')
            sentences = text.split('. ')
            
            for sent in sentences:
                if any(kw in sent.lower() for kw in exp_keywords):
                    sent = sent.strip()
                    if 30 < len(sent) < 250:
                        sent = sent[0].upper() + sent[1:] if sent else sent
                        if not sent.endswith('.'):
                            sent += '.'
                        explanations.append((sent, i+1))
        
        return explanations[:5]
    
    def _extract_relevant_sentences(self, question: str, contexts: list) -> list:
        """Extract sentences most relevant to the question."""
        relevant = []
        
        # Extract key terms from question
        question_terms = set(question.lower().split())
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'do', 'does', 'can', 'are', 'of', 'to', 'in', 'for', 'on', 'with'}
        question_terms = question_terms - stop_words
        
        for i, c in enumerate(contexts[:5]):
            if not c:
                continue
            text = c.get('text', '')
            sentences = text.split('. ')
            
            for sent in sentences:
                sent_lower = sent.lower()
                # Count matching terms
                matches = sum(1 for term in question_terms if term in sent_lower)
                
                if matches >= 1 and 30 < len(sent) < 300:
                    sent = sent.strip()
                    sent = sent[0].upper() + sent[1:] if sent else sent
                    if not sent.endswith('.'):
                        sent += '.'
                    relevant.append((sent, i+1, matches))
        
        # Sort by number of matches
        relevant.sort(key=lambda x: x[2], reverse=True)
        return [(sent, idx) for sent, idx, _ in relevant[:5]]
    
    def _extract_subject(self, question: str) -> str:
        """Extract the main subject from a question."""
        question = question.lower()
        for phrase in ['what is', 'define', 'explain']:
            if phrase in question:
                subject = question.split(phrase)[-1].strip().rstrip('?')
                return subject
        return ''
    
    def _fallback_answer(self, question: str) -> str:
        """Provide a fallback answer when context doesn't help."""
        lines = []
        lines.append("I found some relevant literature, but couldn't extract a clear answer to your specific question.")
        lines.append("\n**Suggestion:** Try rephrasing your question or check if:")
        lines.append("- Your papers contain information about this topic")
        lines.append("- The question is specific enough")
        lines.append("- You have enough papers indexed")
        lines.append("\nFor better results, consider setting the HF_MODEL environment variable to use a real language model.")
        return "\n".join(lines)

    def _mock_copilot(self, question: str, contexts: List[dict]) -> str:
        """Generate mock executable code for Co-pilot mode."""
        # Analyze question to determine task
        q_lower = question.lower()
        
        code_lines = ["import scanpy as sc", "import numpy as np", "import pandas as pd", ""]
        
        # Add context as comments
        if contexts:
            code_lines.append("# Based on scientific literature:")
            for i, c in enumerate(contexts[:3]):
                if c:
                    source = c.get('source', 'Unknown')
                    code_lines.append(f"# [{i+1}] {source}")
            code_lines.append("")
        
        # Generate appropriate code based on question keywords
        if any(kw in q_lower for kw in ['marker', 'genes', 'differential', 'de']):
            code_lines.extend([
                "# Find marker genes for clusters",
                "# Method: Wilcoxon rank-sum test (recommended in [1])",
                "sc.tl.rank_genes_groups(",
                "    adata,",
                "    groupby='leiden',  # or 'louvain', or your cluster column",
                "    method='wilcoxon',  # alternatives: 't-test', 'logreg'",
                "    n_genes=50",
                ")",
                "",
                "# Visualize top marker genes",
                "sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)",
                "",
                "# Get marker genes as DataFrame",
                "marker_genes = sc.get.rank_genes_groups_df(adata, group='0')",
                "print(marker_genes.head(10))"
            ])
        
        elif any(kw in q_lower for kw in ['cluster', 'leiden', 'louvain']):
            code_lines.extend([
                "# Clustering analysis",
                "# Ensure neighbors are computed (required for clustering) [1]",
                "if 'neighbors' not in adata.uns:",
                "    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)",
                "",
                "# Leiden clustering (recommended over Louvain) [2]",
                "sc.tl.leiden(adata, resolution=0.5)  # adjust resolution as needed",
                "",
                "# Visualize clusters on UMAP",
                "sc.pl.umap(adata, color=['leiden'], legend_loc='on data')",
                "",
                "# Quality metrics per cluster",
                "cluster_sizes = adata.obs['leiden'].value_counts()",
                "print('Cluster sizes:', cluster_sizes)"
            ])
        
        elif any(kw in q_lower for kw in ['trajectory', 'pseudotime', 'temporal', 'dpt']):
            code_lines.extend([
                "# Pseudo-temporal ordering analysis",
                "# Using Diffusion Pseudotime (DPT) method [1]",
                "",
                "# Compute diffusion map",
                "sc.tl.diffmap(adata)",
                "",
                "# Set root cell (modify index as appropriate)",
                "root_cell_idx = 0  # Change to your root cell",
                "adata.uns['iroot'] = root_cell_idx",
                "",
                "# Compute DPT",
                "sc.tl.dpt(adata)",
                "",
                "# Visualize pseudotime",
                "sc.pl.umap(adata, color=['dpt_pseudotime'], color_map='viridis')",
                "",
                "# Alternative: PAGA trajectory inference [2]",
                "# sc.tl.paga(adata, groups='leiden')",
                "# sc.pl.paga(adata, color=['leiden'])"
            ])
        
        elif any(kw in q_lower for kw in ['integrate', 'batch', 'harmony', 'combat']):
            code_lines.extend([
                "# Batch correction / Data integration",
                "# Using Harmony (requires harmonypy package) [1]",
                "",
                "# Option 1: Harmony (recommended for large datasets)",
                "# import harmonypy",
                "# sc.external.pp.harmony_integrate(adata, 'batch')  # 'batch' is your batch column",
                "",
                "# Option 2: Combat (built-in scanpy)",
                "sc.pp.combat(adata, key='batch')  # 'batch' is your batch column",
                "",
                "# Re-run dimensionality reduction after integration",
                "sc.pp.neighbors(adata)",
                "sc.tl.umap(adata)",
                "",
                "# Visualize by batch",
                "sc.pl.umap(adata, color=['batch'], title='After batch correction')"
            ])
        
        elif any(kw in q_lower for kw in ['qc', 'quality', 'filter']):
            code_lines.extend([
                "# Quality control for scRNA-seq data [1]",
                "",
                "# Calculate QC metrics",
                "sc.pp.calculate_qc_metrics(",
                "    adata,",
                "    qc_vars=['mt'],  # mitochondrial genes (assumes var_names contain 'MT-')",
                "    percent_top=None,",
                "    log1p=False,",
                "    inplace=True",
                ")",
                "",
                "# Visualize QC metrics",
                "sc.pl.violin(",
                "    adata,",
                "    ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],",
                "    jitter=0.4,",
                "    multi_panel=True",
                ")",
                "",
                "# Filter cells [2]",
                "# Adjust thresholds based on your data",
                "sc.pp.filter_cells(adata, min_genes=200)",
                "sc.pp.filter_genes(adata, min_cells=3)",
                "",
                "# Remove cells with high mitochondrial content",
                "adata = adata[adata.obs['pct_counts_mt'] < 5, :]",
                "",
                "print(f'Filtered data: {adata.n_obs} cells x {adata.n_vars} genes')"
            ])
        
        elif any(kw in q_lower for kw in ['normalize', 'normalization']):
            code_lines.extend([
                "# Normalization of scRNA-seq data",
                "# Standard log-normalization (recommended in [1])",
                "",
                "# Normalize to 10,000 counts per cell",
                "sc.pp.normalize_total(adata, target_sum=1e4)",
                "",
                "# Log-transform",
                "sc.pp.log1p(adata)",
                "",
                "# Alternative: SCTransform-style normalization [2]",
                "# Requires scvi-tools package",
                "# import scvi",
                "# scvi.model.SCVI.setup_anndata(adata)",
                "# model = scvi.model.SCVI(adata)",
                "# model.train()",
                "",
                "print('Normalization complete')"
            ])
        
        elif any(kw in q_lower for kw in ['hvg', 'variable', 'feature']):
            code_lines.extend([
                "# Highly variable genes (HVG) selection [1]",
                "",
                "# Identify highly variable genes",
                "sc.pp.highly_variable_genes(",
                "    adata,",
                "    n_top_genes=2000,",
                "    flavor='seurat_v3'  # alternatives: 'seurat', 'cell_ranger'",
                ")",
                "",
                "# Subset to HVGs (optional but recommended)",
                "# adata = adata[:, adata.var['highly_variable']]",
                "",
                "# Visualize HVG selection",
                "sc.pl.highly_variable_genes(adata)",
                "",
                "n_hvg = adata.var['highly_variable'].sum()",
                "print(f'Found {n_hvg} highly variable genes')"
            ])
        
        elif any(kw in q_lower for kw in ['umap', 'tsne', 'visualization', 'visualize']):
            code_lines.extend([
                "# Dimensionality reduction and visualization",
                "",
                "# Run PCA (if not already done) [1]",
                "if 'X_pca' not in adata.obsm:",
                "    sc.pp.pca(adata, n_comps=50)",
                "",
                "# Compute neighborhood graph [1]",
                "if 'neighbors' not in adata.uns:",
                "    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)",
                "",
                "# Compute UMAP [2]",
                "sc.tl.umap(adata)",
                "",
                "# Visualize UMAP",
                "sc.pl.umap(adata, color=['leiden'], legend_loc='on data')",
                "",
                "# Alternative: t-SNE",
                "# sc.tl.tsne(adata, n_pcs=40)",
                "# sc.pl.tsne(adata, color=['leiden'])"
            ])
        
        elif any(kw in q_lower for kw in ['annotate', 'annotation', 'cell type']):
            code_lines.extend([
                "# Cell type annotation",
                "# Using marker genes (manual annotation) [1]",
                "",
                "# Define marker genes for known cell types",
                "marker_genes = {",
                "    'T cells': ['CD3D', 'CD3E', 'CD8A'],",
                "    'B cells': ['CD79A', 'MS4A1'],",
                "    'Monocytes': ['CD14', 'FCGR3A'],",
                "    'NK cells': ['GNLY', 'NKG7'],",
                "    # Add more cell types and markers",
                "}",
                "",
                "# Score cells for each cell type",
                "for cell_type, markers in marker_genes.items():",
                "    sc.tl.score_genes(adata, markers, score_name=f'{cell_type}_score')",
                "",
                "# Visualize marker expression",
                "sc.pl.umap(adata, color=list(marker_genes.keys()) + ['_score'], ncols=2)",
                "",
                "# Alternative: Automated annotation with celltypist [2]",
                "# import celltypist",
                "# model = celltypist.models.Model.load(model='Immune_All_Low.pkl')",
                "# predictions = celltypist.annotate(adata, model=model)",
                "# adata.obs['cell_type'] = predictions.predicted_labels"
            ])
        
        else:
            # Generic analysis pipeline
            code_lines.extend([
                "# Standard scRNA-seq analysis pipeline [1, 2]",
                "",
                "# 1. Quality control (if not done)",
                "if 'n_genes_by_counts' not in adata.obs:",
                "    sc.pp.calculate_qc_metrics(adata, inplace=True)",
                "",
                "# 2. Normalization",
                "sc.pp.normalize_total(adata, target_sum=1e4)",
                "sc.pp.log1p(adata)",
                "",
                "# 3. Feature selection",
                "sc.pp.highly_variable_genes(adata, n_top_genes=2000)",
                "",
                "# 4. Dimensionality reduction",
                "sc.pp.pca(adata, n_comps=50)",
                "sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)",
                "sc.tl.umap(adata)",
                "",
                "# 5. Clustering",
                "sc.tl.leiden(adata, resolution=0.5)",
                "",
                "# 6. Visualization",
                "sc.pl.umap(adata, color=['leiden'], legend_loc='on data')",
                "",
                "# 7. Find marker genes",
                "sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon')",
                "sc.pl.rank_genes_groups(adata, n_genes=20)",
                "",
                "print('Analysis pipeline complete!')"
            ])
        
        code_lines.append("")
        code_lines.append("# Note: Adjust parameters based on your specific dataset and research question")
        code_lines.append("# Citations [n] refer to the scientific papers retrieved above")
        
        return "\n".join(code_lines)


class RAGChain:
    def __init__(self, embedder: Embedder = None, faiss_store: FaissStore = None):
        self.embedder = embedder
        self.faiss = faiss_store
        self.llm = LLMWrapper()

    def answer(self, question: str, k: int = 5, mode: str = "qa") -> Tuple[str, List[dict], List[float]]:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            mode: Generation mode - "qa" for Q&A, "copilot" for code generation
            
        Returns:
            Tuple of (answer, retrieved_docs, distances)
        """
        if self.embedder is None or self.faiss is None:
            raise RuntimeError("Embedder and FaissStore must be initialized. Load the index first.")
        
        # Embed the question
        qvec = self.embedder.embed_texts([question])[0]
        
        # Retrieve relevant documents
        hits, dists = self.faiss.search(qvec, k=k)
        
        # Generate answer or code
        answer = self.llm.generate(question, hits, mode=mode)
        
        return answer, hits, dists

    def rebuild_index_from_papers(
        self,
        papers_folder: str = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        batch_size: int = 16
    ) -> int:
        """
        Rebuild the FAISS index from papers in the specified folder.
        
        Args:
            papers_folder: Path to folder containing papers
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            batch_size: Batch size for embedding
            
        Returns:
            Number of chunks indexed
        """
        if papers_folder is None:
            papers_folder = os.path.join("scRAG", "data", "papers")
        
        print(f"Loading documents from {papers_folder}...")
        docs = load_documents_from_papers_folder(papers_folder)
        
        if not docs:
            print("No documents found!")
            return 0
        
        print(f"Found {len(docs)} documents. Chunking...")
        chunks = docs_to_chunks(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"Created {len(chunks)} chunks.")
        
        texts = [c["text"] for c in chunks]
        
        # Initialize embedder if needed
        if self.embedder is None:
            print("Initializing embedder...")
            self.embedder = Embedder()
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.embed_texts(texts, batch_size=batch_size)
        dim = embeddings.shape[1]
        print(f"Generated embeddings with dimension {dim}")
        
        # Initialize or reset FAISS store
        if self.faiss is None:
            print("Initializing FAISS store...")
            self.faiss = FaissStore(dim=dim)
        else:
            print("Resetting existing FAISS store...")
            self.faiss.reset()
            self.faiss._init_index(dim)
        
        # Prepare metadata
        metas = [
            {
                "id": c["id"],
                "source": c["source"],
                "source_path": c.get("source_path"),
                "excerpt": c.get("excerpt"),
                "text": c.get("text")
            }
            for c in chunks
        ]
        
        # Add to FAISS
        print("Adding embeddings to FAISS index...")
        self.faiss.add(embeddings, metas)
        
        # Save index
        print("Saving index...")
        self.faiss.save()
        print(f"✓ Index saved successfully with {len(chunks)} chunks!")
        
        return len(chunks)
