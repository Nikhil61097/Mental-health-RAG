# 🧠 Mental Health FAQ RAG — Local Python App

A 100% local Retrieval-Augmented Generation (RAG) chatbot for answering mental health FAQ questions — **no API keys, no cloud LLMs, runs fully on your computer using open-source models!**

---

## Features

- **Works completely offline:** No internet needed after first model download.
- **Retrieves top FAQ answers** from a customizable mental health dataset (CSV).
- **Uses local LLMs** (Phi-3 mini, Mistral, etc.) for context-aware, generative answers.
- **GPU-accelerated:** Fast on RTX 3070 Ti and similar GPUs.
- **Privacy-first:** No data leaves your machine.

---

## Quickstart

### 1. **Clone the repository**
### 2. **Download this file or use the csv in the repo https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot***
```bash
git clone https://github.com/Nikhil61097/Mental-health-RAG.git
cd Mental-health-RAG
pip install pandas numpy sentence-transformers transformers torch torchvision
