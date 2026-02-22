# 🧠 Fully Local RAG System (Phi3 + FAISS)

A production-style Retrieval-Augmented Generation (RAG) system built using:

- 🟢 Phi-3 (local LLM via Ollama)
- 🟢 HuggingFace Sentence Transformers (local embeddings)
- 🟢 FAISS (vector similarity search)
- 🟢 LangChain (orchestration layer)

Optimized for low-resource systems (8GB RAM).

---

## 🚀 Problem Statement

Large Language Models hallucinate when asked questions outside their training data.

This project solves that problem using Retrieval-Augmented Generation (RAG):

Instead of relying only on model memory,
we retrieve relevant documents and ground the response.

---

## 🏗️ Architecture

User Query  
↓  
Embedding (all-MiniLM-L6-v2)  
↓  
FAISS Vector Search (Top-K Retrieval)  
↓  
Phi-3 Local LLM (Ollama)  
↓  
Grounded Answer  

---

## 🔬 Key Engineering Decisions

### ✅ Why Local Embeddings?
- Avoid API cost
- Faster inference
- No dependency on external services

### ✅ Why Phi-3?
- Lightweight (works on 8GB RAM)
- Good reasoning performance
- Efficient inference

### ✅ Why FAISS?
- Efficient Approximate Nearest Neighbor (ANN) search
- Scales to large vector datasets

---

## 📦 Features

- Document chunking with overlap
- Semantic vector search
- Top-K retrieval
- Hallucination reduction via grounded prompts
- Fully offline execution
- Optimized for resource-constrained hardware

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
ollama pull phi3
python app.py
