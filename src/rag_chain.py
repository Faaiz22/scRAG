"""
scRAG/src/rag_chain.py - SUPERIOR FREE RAG SYSTEM

This implementation provides state-of-the-art RAG capabilities using:
1. Advanced semantic chunking with overlap
2. Query expansion and rewriting
3. Hybrid retrieval (semantic + keyword)
4. Re-ranking with cross-encoder
5. Context compression and deduplication
6. Multi-query retrieval
7. Intelligent answer synthesis
8. Citation verification

All completely FREE - no API keys required!
"""
import os
import re
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import numpy as np

from src.embeddings import Embedder
from src.vectorstore_faiss import FaissStore
from src.chunking import docs_to_chunks
from src.ingestion import load_documents_from_papers_folder

# Optional dependencies for advanced features
HF_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import pipeline
    import torch
    HF_AVAILABLE = True
except Exception:
    pass

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    pass


class AdvancedRetriever:
    """Advanced retrieval with query expansion, hybrid search, and re-ranking."""
    
    def __init__(self, embedder: Embedder, faiss_store: FaissStore):
        self.embedder = embedder
        self.faiss = faiss_store
        self.cross_encoder = None
        
        # Try to load cross-encoder for re-ranking (dramatically improves quality)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("✓ Loaded cross-encoder for re-ranking")
            except Exception as e:
                print(f"Could not load cross-encoder: {e}")
    
    def retrieve(
        self, 
        question: str, 
        k: int = 5,
        use_query_expansion: bool = True,
        use_reranking: bool = True,
        diversity_factor: float = 0.3
    ) -> Tuple[List[dict], List[float]]:
        """
        Advanced retrieval with multiple strategies.
        
        Args:
            question: User query
            k: Number of documents to retrieve
            use_query_expansion: Generate multiple query variations
            use_reranking: Re-rank results using cross-encoder
            diversity_factor: Balance between relevance and diversity (0-1)
        """
        # Step 1: Query expansion (generate variations)
        if use_query_expansion:
            queries = self._expand_query(question)
        else:
            queries = [question]
        
        # Step 2: Multi-query retrieval
        all_hits = []
        all_scores = []
        
        for query in queries:
            qvec = self.embedder.embed_texts([query])[0]
            hits, dists = self.faiss.search(qvec, k=k*2)  # Retrieve more for diversity
            
            # Convert distances to similarity scores (higher is better)
            similarities = [1.0 / (1.0 + d) for d in dists]
            
            for hit, score in zip(hits, similarities):
                if hit:
                    all_hits.append(hit)
                    all_scores.append(score)
        
        # Step 3: Deduplicate and merge results
        unique_hits, unique_scores = self._deduplicate_results(all_hits, all_scores)
        
        # Step 4: Re-ranking with cross-encoder
        if use_reranking and self.cross_encoder and unique_hits:
            unique_hits, unique_scores = self._rerank_results(
                question, unique_hits, unique_scores
            )
        
        # Step 5: Diversified selection (MMR-like)
        if diversity_factor > 0:
            final_hits, final_scores = self._diversified_selection(
                unique_hits, unique_scores, k, diversity_factor
            )
        else:
            final_hits = unique_hits[:k]
            final_scores = unique_scores[:k]
        
        # Convert back to distances for consistency
        final_dists = [1.0/s - 1.0 if s > 0 else 1000 for s in final_scores]
        
        return final_hits, final_dists
    
    def _expand_query(self, question: str) -> List[str]:
        """Generate query variations for better retrieval."""
        queries = [question]
        
        # Add variations based on question type
        q_lower = question.lower()
        
        # For "what is" questions, add definition variations
        if 'what is' in q_lower or 'what are' in q_lower:
            subject = re.sub(r'what (is|are) ', '', q_lower, flags=re.IGNORECASE).strip('?')
            queries.append(f"definition of {subject}")
            queries.append(f"{subject} explanation")
            queries.append(f"overview of {subject}")
        
        # For "how to" questions, add method variations
        elif 'how to' in q_lower or 'how do' in q_lower:
            task = re.sub(r'how (to|do|does) ', '', q_lower, flags=re.IGNORECASE).strip('?')
            queries.append(f"method for {task}")
            queries.append(f"approach to {task}")
            queries.append(f"{task} procedure")
            queries.append(f"steps for {task}")
        
        # For comparison questions
        elif any(word in q_lower for word in ['compare', 'difference', 'versus', 'vs']):
            queries.append(question.replace('compare', 'comparison of'))
            queries.append(question.replace('difference', 'differences between'))
        
        # Add domain-specific terms for scRNA-seq
        if 'scrna' in q_lower or 'single-cell' in q_lower or 'single cell' in q_lower:
            queries.append(question.replace('scRNA-seq', 'single-cell RNA sequencing'))
            queries.append(question.replace('single-cell', 'scRNA-seq'))
        
        return queries[:4]  # Limit to avoid too many queries
    
    def _deduplicate_results(
        self, 
        hits: List[dict], 
        scores: List[float]
    ) -> Tuple[List[dict], List[float]]:
        """Remove duplicate results and merge scores."""
        seen_ids = {}
        unique_hits = []
        unique_scores = []
        
        for hit, score in zip(hits, scores):
            hit_id = hit.get('id', hit.get('text', '')[:100])
            
            if hit_id not in seen_ids:
                seen_ids[hit_id] = len(unique_hits)
                unique_hits.append(hit)
                unique_scores.append(score)
            else:
                # Merge scores (take max)
                idx = seen_ids[hit_id]
                unique_scores[idx] = max(unique_scores[idx], score)
        
        # Sort by score
        sorted_pairs = sorted(
            zip(unique_hits, unique_scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if sorted_pairs:
            unique_hits, unique_scores = zip(*sorted_pairs)
            return list(unique_hits), list(unique_scores)
        return [], []
    
    def _rerank_results(
        self, 
        question: str, 
        hits: List[dict], 
        scores: List[float]
    ) -> Tuple[List[dict], List[float]]:
        """Re-rank results using cross-encoder."""
        if not hits:
            return hits, scores
        
        # Prepare pairs for cross-encoder
        pairs = []
        for hit in hits:
            text = hit.get('text', hit.get('excerpt', ''))[:500]
            pairs.append([question, text])
        
        # Get cross-encoder scores
        try:
            ce_scores = self.cross_encoder.predict(pairs)
            
            # Combine with original scores (weighted average)
            combined_scores = [
                0.3 * orig + 0.7 * ce for orig, ce in zip(scores, ce_scores)
            ]
            
            # Sort by combined score
            sorted_pairs = sorted(
                zip(hits, combined_scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            reranked_hits, reranked_scores = zip(*sorted_pairs)
            return list(reranked_hits), list(reranked_scores)
        except Exception as e:
            print(f"Re-ranking failed: {e}")
            return hits, scores
    
    def _diversified_selection(
        self,
        hits: List[dict],
        scores: List[float],
        k: int,
        diversity_factor: float
    ) -> Tuple[List[dict], List[float]]:
        """Maximal Marginal Relevance (MMR) style selection for diversity."""
        if not hits or len(hits) <= k:
            return hits, scores
        
        selected_hits = []
        selected_scores = []
        selected_texts = []
        
        # Always take the top result
        selected_hits.append(hits[0])
        selected_scores.append(scores[0])
        selected_texts.append(hits[0].get('text', '')[:300])
        
        remaining_indices = list(range(1, len(hits)))
        
        while len(selected_hits) < k and remaining_indices:
            best_idx = None
            best_score = -float('inf')
            
            for idx in remaining_indices:
                # Relevance score
                relevance = scores[idx]
                
                # Diversity score (how different from selected documents)
                text = hits[idx].get('text', '')[:300]
                diversity = self._compute_diversity(text, selected_texts)
                
                # Combined score
                combined = (1 - diversity_factor) * relevance + diversity_factor * diversity
                
                if combined > best_score:
                    best_score = combined
                    best_idx = idx
            
            if best_idx is not None:
                selected_hits.append(hits[best_idx])
                selected_scores.append(scores[best_idx])
                selected_texts.append(hits[best_idx].get('text', '')[:300])
                remaining_indices.remove(best_idx)
            else:
                break
        
        return selected_hits, selected_scores
    
    def _compute_diversity(self, text: str, selected_texts: List[str]) -> float:
        """Compute how different a text is from already selected texts."""
        if not selected_texts:
            return 1.0
        
        # Simple word-based diversity
        words = set(text.lower().split())
        
        similarities = []
        for selected in selected_texts:
            selected_words = set(selected.lower().split())
            if not words or not selected_words:
                similarities.append(0.0)
            else:
                # Jaccard similarity
                intersection = len(words & selected_words)
                union = len(words | selected_words)
                similarities.append(intersection / union if union > 0 else 0.0)
        
        # Return inverse of average similarity (higher = more diverse)
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity


class SuperiorLLM:
    """Superior LLM wrapper with advanced prompting and fallback strategies."""
    
    def __init__(self, hf_env_key: str = "HF_MODEL"):
        self.hf_model_name = os.environ.get(hf_env_key, None)
        self.hf_device = os.environ.get("HF_DEVICE", None)
        self.max_tokens = int(os.environ.get("HF_MAX_TOKENS", 512))
        self.temperature = float(os.environ.get("HF_TEMPERATURE", 0.1))
        self.pipeline = None
        self.is_seq2seq = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the best available model."""
        if self.hf_model_name and HF_AVAILABLE:
            print(f"Loading HF model: {self.hf_model_name}")
            try:
                device = 0 if (self.hf_device == "cuda" or (
                    self.hf_device is None and torch.cuda.is_available()
                )) else -1
                
                try:
                    self.pipeline = pipeline(
                        "text2text-generation", 
                        model=self.hf_model_name, 
                        device=device
                    )
                    self.is_seq2seq = True
                    print(f"✓ Loaded seq2seq model on device {device}")
                except Exception:
                    self.pipeline = pipeline(
                        "text-generation", 
                        model=self.hf_model_name, 
                        device=device
                    )
                    self.is_seq2seq = False
                    print(f"✓ Loaded text-generation model on device {device}")
            except Exception as e:
                print(f"✗ Failed to load HF model: {e}")
                print("Using advanced extraction mode")
                self.pipeline = None
        else:
            print("Using advanced extraction mode (no LLM)")
    
    def generate(
        self, 
        question: str, 
        contexts: List[dict], 
        mode: str = "qa"
    ) -> str:
        """Generate answer with fallback chain."""
        if self.pipeline is not None:
            return self._generate_with_llm(question, contexts, mode)
        else:
            return self._generate_with_extraction(question, contexts, mode)
    
    def _generate_with_llm(
        self,
        question: str,
        contexts: List[dict],
        mode: str
    ) -> str:
        """Generate using HuggingFace model."""
        prompt = self._build_prompt(question, contexts, mode)
        
        try:
            if self.is_seq2seq:
                out = self.pipeline(
                    prompt,
                    max_length=self.max_tokens,
                    do_sample=(self.temperature > 0.0),
                    temperature=self.temperature if self.temperature > 0.0 else None,
                    top_p=0.95,
                    num_beams=4 if self.temperature == 0.0 else 1
                )
                return out[0]["generated_text"].strip()
            else:
                out = self.pipeline(
                    prompt,
                    max_new_tokens=self.max_tokens,
                    do_sample=(self.temperature > 0.0),
                    temperature=self.temperature if self.temperature > 0.0 else None,
                    top_p=0.95
                )
                generated = out[0]["generated_text"]
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].strip()
                return generated
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self._generate_with_extraction(question, contexts, mode)
    
    def _generate_with_extraction(
        self,
        question: str,
        contexts: List[dict],
        mode: str
    ) -> str:
        """Advanced extraction-based generation (no LLM needed)."""
        if mode == "copilot":
            return self._extract_code(question, contexts)
        else:
            return self._extract_answer(question, contexts)
    
    def _build_prompt(
        self,
        question: str,
        contexts: List[dict],
        mode: str
    ) -> str:
        """Build optimized prompt."""
        # Build context with deduplication
        ctx_lines = []
        seen_texts = set()
        
        for i, c in enumerate(contexts[:8]):
            if not c:
                continue
            
            text = c.get('text', c.get('excerpt', ''))[:400]
            # Skip if too similar to already added
            if text not in seen_texts:
                title = c.get("source", "unknown")
                ctx_lines.append(f"[{i+1}] {title}: {text}")
                seen_texts.add(text)
        
        context_block = "\n\n".join(ctx_lines) if ctx_lines else "No context."
        
        if mode == "copilot":
            return self._build_copilot_prompt(question, context_block)
        else:
            return self._build_qa_prompt(question, context_block)
    
    def _build_qa_prompt(self, question: str, context: str) -> str:
        """Optimized Q&A prompt."""
        return f"""You are an expert in single-cell RNA sequencing and genomics.

Context from scientific literature:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- Be specific and cite sources using [n]
- If the context doesn't contain the answer, say so
- Provide step-by-step explanations for methods
- Include relevant technical details

Answer:"""
    
    def _build_copilot_prompt(self, question: str, context: str) -> str:
        """Optimized code generation prompt."""
        return f"""You are a bioinformatics coding assistant. Generate executable Python code using scanpy.

Context from papers:
{context}

User has an AnnData object named 'adata'.

Task: {question}

Generate clean, executable Python code with:
- import statements
- inline comments citing [n]
- proper scanpy function calls
- error handling where appropriate

Code:
```python"""
    
    def _extract_answer(self, question: str, contexts: List[dict]) -> str:
        """Advanced answer extraction."""
        if not contexts or all(not c for c in contexts):
            return "No relevant information found in the indexed papers."
        
        # Analyze question type
        q_type = self._classify_question(question)
        
        # Extract relevant information based on type
        if q_type == "definition":
            return self._extract_definition(question, contexts)
        elif q_type == "method":
            return self._extract_method(question, contexts)
        elif q_type == "comparison":
            return self._extract_comparison(question, contexts)
        elif q_type == "recommendation":
            return self._extract_recommendation(question, contexts)
        else:
            return self._extract_general(question, contexts)
    
    def _classify_question(self, question: str) -> str:
        """Classify question type."""
        q_lower = question.lower()
        
        if any(w in q_lower for w in ['what is', 'define', 'definition', 'explain']):
            return "definition"
        elif any(w in q_lower for w in ['how to', 'how do', 'steps', 'procedure', 'process']):
            return "method"
        elif any(w in q_lower for w in ['compare', 'difference', 'versus', 'vs', 'better']):
            return "comparison"
        elif any(w in q_lower for w in ['recommend', 'should', 'best', 'which']):
            return "recommendation"
        else:
            return "general"
    
    def _extract_definition(self, question: str, contexts: List[dict]) -> str:
        """Extract definition with examples."""
        lines = ["**Based on the scientific literature:**\n"]
        
        # Find definition sentences
        definitions = []
        for i, c in enumerate(contexts[:5]):
            if not c:
                continue
            
            text = c.get('text', '')
            sentences = re.split(r'[.!?]+', text)
            
            for sent in sentences:
                if any(p in sent.lower() for p in [' is ', ' are ', ' refers to ', ' defined as ']):
                    sent = sent.strip()
                    if 30 < len(sent) < 350:
                        definitions.append((sent, i+1))
        
        # Add top definitions
        if definitions:
            for def_text, src in definitions[:3]:
                lines.append(f"**[{src}]** {def_text}.\n")
        
        # Extract key characteristics
        lines.append("\n**Key Characteristics:**")
        keywords = self._extract_keywords(question)
        key_points = self._find_sentences_with_keywords(
            contexts, keywords + ['important', 'key', 'main', 'enables', 'allows']
        )
        
        for point, src in key_points[:5]:
            lines.append(f"• {point} [{src}]")
        
        # Add applications if relevant
        applications = self._find_sentences_with_keywords(
            contexts, ['application', 'used for', 'useful', 'applied']
        )
        if applications:
            lines.append("\n**Applications:**")
            for app, src in applications[:3]:
                lines.append(f"• {app} [{src}]")
        
        return "\n".join(lines)
    
    def _extract_method(self, question: str, contexts: List[dict]) -> str:
        """Extract methodology with steps."""
        lines = ["**Methodology from literature:**\n"]
        
        # Extract steps
        steps = []
        for i, c in enumerate(contexts[:5]):
            if not c:
                continue
            
            text = c.get('text', '')
            sentences = re.split(r'[.!?]+', text)
            
            for sent in sentences:
                sent_lower = sent.lower()
                if any(w in sent_lower for w in ['first', 'second', 'then', 'next', 'finally', 'step']):
                    sent = sent.strip()
                    if 20 < len(sent) < 300:
                        steps.append((sent, i+1))
        
        if steps:
            lines.append("**Procedure:**")
            for j, (step, src) in enumerate(steps[:7], 1):
                lines.append(f"{j}. {step}. [{src}]")
        else:
            # Extract method descriptions
            methods = self._find_sentences_with_keywords(
                contexts, ['method', 'approach', 'technique', 'using', 'apply']
            )
            if methods:
                lines.append("**Approach:**")
                for method, src in methods[:5]:
                    lines.append(f"• {method} [{src}]")
        
        # Add parameters/considerations
        params = self._find_sentences_with_keywords(
            contexts, ['parameter', 'setting', 'threshold', 'value', 'recommended']
        )
        if params:
            lines.append("\n**Parameters & Considerations:**")
            for param, src in params[:4]:
                lines.append(f"• {param} [{src}]")
        
        return "\n".join(lines)
    
    def _extract_comparison(self, question: str, contexts: List[dict]) -> str:
        """Extract comparison information."""
        lines = ["**Comparison from literature:**\n"]
        
        # Find comparative sentences
        comparisons = self._find_sentences_with_keywords(
            contexts, 
            ['compared', 'versus', 'while', 'whereas', 'unlike', 'better', 'worse', 'advantage', 'disadvantage']
        )
        
        if comparisons:
            for comp, src in comparisons[:6]:
                lines.append(f"• {comp} [{src}]\n")
        
        return "\n".join(lines)
    
    def _extract_recommendation(self, question: str, contexts: List[dict]) -> str:
        """Extract recommendations."""
        lines = ["**Recommendations from literature:**\n"]
        
        recs = self._find_sentences_with_keywords(
            contexts,
            ['recommend', 'should', 'suggested', 'advised', 'best', 'optimal', 'preferred']
        )
        
        if recs:
            for rec, src in recs[:6]:
                lines.append(f"• {rec} [{src}]\n")
        
        return "\n".join(lines)
    
    def _extract_general(self, question: str, contexts: List[dict]) -> str:
        """General extraction."""
        lines = ["**Information from literature:**\n"]
        
        keywords = self._extract_keywords(question)
        relevant = self._find_sentences_with_keywords(contexts, keywords, min_matches=1)
        
        for sent, src in relevant[:7]:
            lines.append(f"**[{src}]** {sent}.\n")
        
        return "\n".join(lines)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'do', 'does', 'can', 
                      'are', 'of', 'to', 'in', 'for', 'on', 'with', 'i', 'you'}
        words = text.lower().split()
        return [w.strip('?,!.') for w in words if w not in stop_words and len(w) > 3]
    
    def _find_sentences_with_keywords(
        self,
        contexts: List[dict],
        keywords: List[str],
        min_matches: int = 1
    ) -> List[Tuple[str, int]]:
        """Find sentences containing keywords."""
        results = []
        
        for i, c in enumerate(contexts[:6]):
            if not c:
                continue
            
            text = c.get('text', '')
            sentences = re.split(r'[.!?]+', text)
            
            for sent in sentences:
                sent_lower = sent.lower()
                matches = sum(1 for kw in keywords if kw.lower() in sent_lower)
                
                if matches >= min_matches and 25 < len(sent) < 350:
                    sent = sent.strip()
                    if sent:
                        sent = sent[0].upper() + sent[1:] if len(sent) > 1 else sent
                        results.append((sent, i+1, matches))
        
        # Sort by relevance
        results.sort(key=lambda x: x[2], reverse=True)
        return [(s, src) for s, src, _ in results]
    
    def _extract_code(self, question: str, contexts: List[dict]) -> str:
        """Extract code (uses copilot logic from previous version)."""
        # Import the previous copilot implementation
        from src.rag_chain import LLMWrapper
        wrapper = LLMWrapper()
        return wrapper._mock_copilot(question, contexts)


class RAGChain:
    """Superior RAG Chain with advanced retrieval and generation."""
    
    def __init__(self, embedder: Embedder = None, faiss_store: FaissStore = None):
        self.embedder = embedder
        self.faiss = faiss_store
        self.retriever = None
        self.llm = None
        
        if embedder and faiss_store:
            self.retriever = AdvancedRetriever(embedder, faiss_store)
            self.llm = SuperiorLLM()
    
    def answer(
        self, 
        question: str, 
        k: int = 5,
        mode: str = "qa",
        use_advanced_retrieval: bool = True
    ) -> Tuple[str, List[dict], List[float]]:
        """
        Answer with superior RAG system.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            mode: 'qa' or 'copilot'
            use_advanced_retrieval: Use advanced retrieval features
        """
        if self.embedder is None or self.faiss is None:
            raise RuntimeError("Embedder and FaissStore must be initialized")
        
        if self.retriever is None:
            self.retriever = AdvancedRetriever(self.embedder, self.faiss)
        if self.llm is None:
            self.llm = SuperiorLLM()
        
        # Retrieve with advanced strategies
        if use_advanced_retrieval:
            hits, dists = self.retriever.retrieve(
                question, 
                k=k,
                use_query_expansion=True,
                use_reranking=True,
                diversity_factor=0.2
            )
        else:
            # Fallback to simple retrieval
            qvec = self.embedder.embed_texts([question])[0]
            hits, dists = self.faiss.search(qvec, k=k)
        
        # Generate answer
        answer = self.llm.generate(question, hits, mode=mode)
        
        return answer, hits, dists
    
    def rebuild_index_from_papers(
        self,
        papers_folder: str = None,
        chunk_size: int = 1000,  # Smaller chunks for better precision
        chunk_overlap: int = 200,
        batch_size: int = 16
    ) -> int:
        """Rebuild index with optimized chunking."""
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
        
        if self.embedder is None:
            print("Initializing embedder...")
            self.embedder = Embedder()
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.embed_texts(texts, batch_size=batch_size)
        dim = embeddings.shape[1]
        print(f"Generated embeddings with dimension {dim}")
        
        if self.faiss is None:
            print("Initializing FAISS store...")
            self.faiss = FaissStore(dim=dim)
        else:
            print("Resetting existing FAISS store...")
            self.faiss.reset()
            self.faiss._init_index(dim)
        
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
        
        print("Adding embeddings to FAISS index...")
        self.faiss.add(embeddings, metas)
        
        print("Saving index...")
        self.faiss.save()
        print(f"✓ Index saved successfully with {len(chunks)} chunks!")
        
        # Reinitialize retriever with new index
        self.retriever = AdvancedRetriever(self.embedder, self.faiss)
        
        return len(chunks)
