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

git clone https://github.com/Nikhil61097/Mental-health-RAG.git
cd Mental-health-RAG
pip install pandas numpy sentence-transformers transformers torch torchvision

## Technical Details & Micro Settings

This RAG agent is designed for transparency and reproducibility. Here’s exactly how the pipeline works:

### 1. **FAQ Embedding**
- **Model:** `all-MiniLM-L6-v2` from [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html)
- **Purpose:** Converts each FAQ answer into a dense vector embedding.
- **Batching:** Embeddings are computed for the entire dataset at startup.
- **Storage:** Embeddings are kept in memory (NumPy array) for fast search.

### 2. **Similarity Search (Retriever)**
- **Similarity Metric:** Cosine similarity (implemented via NumPy dot product).
- **Top K:** By default, retrieves the top 3 most similar FAQ answers (`top_k=3`) for every user query.
- **Customization:** You can change `top_k` in the `retrieve()` function for more or fewer context passages.

### 3. **Prompt Construction**
- The user question and the top FAQ answers are concatenated into a single prompt, formatted as:

You are a helpful mental health assistant.
Below are some relevant FAQs. Use these as context to answer the user's question.

Context:
Q: [FAQ Question 1]
A: [FAQ Answer 1]
...
User's question: [User input]



### 4. **LLM Generation**
- **Default Model:** `microsoft/Phi-3-mini-4k-instruct` ([Hugging Face model card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct))
- **Model Type:** Open-source, 3.8B parameter transformer, instruction-tuned
- **Context Window:** Up to 4,000 tokens
- **Device:** Forced to GPU (CUDA, float16) for best speed on RTX 3070 Ti
- **Generation Pipeline:** Hugging Face `pipeline("text-generation", ...)`
- **Parameters:**
- `max_new_tokens=200` (max answer length, can be adjusted)
- `temperature=0.7` (controls randomness/creativity)
- `do_sample=True` (enables sampling for more diverse output)

### 5. **Performance Notes**
- **First run:** Downloads LLM weights (~1.8 GB). After first download, runs fully offline.
- **Typical speed:** On RTX 3070 Ti, answers generated in a few seconds (after model warm-up).
- **RAM/VRAM usage:** Model fits comfortably in 8GB VRAM GPUs.
- **All computation:** Retrieval runs on CPU, LLM inference on GPU.

### 6. **Configurable Parameters**
- **Model selection:** Swap out `MODEL = ...` for any compatible LLM on Hugging Face Hub.
- **FAQ CSV:** Replace `mental_health_faq.csv` with any dataset (columns: `Questions`, `Answers`).
- **top_k:** Change number of retrieved FAQs for RAG context.
- **max_new_tokens:** Change answer verbosity.
- **temperature:** Lower for more factual output; higher for more creative output.

---

**This setup provides a solid, reproducible baseline for RAG research and extension.**  
Feel free to tweak any parameter to suit your use case!


Credits
Phi-3 Mini (Microsoft)

Sentence-Transformers

Transformers

Dataset: Kaggle Mental Health FAQ for Chatbot


MIT License

For education and research. Not a substitute for medical advice.


