import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ------------------------------
# CUDA Availability Check
# ------------------------------
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️  Warning: No CUDA GPU detected! Your model will run on CPU and be very slow.")

# ------------------------------
# 1. Load FAQ Dataset (from CSV)
# ------------------------------
print("Loading FAQ dataset from CSV...")
df = pd.read_csv("mental_health_faq.csv")  # Change this if your filename is different
print(df.head())

# ------------------------------
# 2. Embed Answers
# ------------------------------
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
answers = df["Answers"].tolist()
print("Encoding FAQ answers (this may take a moment)...")
faq_embeddings = embedder.encode(answers, show_progress_bar=True)

# ------------------------------
# 3. Set Up Local LLM (Phi-3 mini, forced to GPU)
# ------------------------------
print("Loading local LLM (Phi-3 mini, this will download ~1.8GB on first run)...")
MODEL = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="cuda",
    torch_dtype=torch.float16
)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
    # <-- Do not specify device=0!
)


# ------------------------------
# 4. Retrieval Function
# ------------------------------
def retrieve(query, top_k=3):
    query_emb = embedder.encode([query])[0]
    scores = np.dot(faq_embeddings, query_emb)
    best_idxs = np.argsort(scores)[::-1][:top_k]
    return df.iloc[best_idxs][["Questions", "Answers"]]

# ------------------------------
# 5. RAG Generation (Local LLM)
# ------------------------------
def rag_answer_local(user_question, retrieved_answers):
    context = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(retrieved_answers['Questions'], retrieved_answers['Answers']))
    prompt = f"""You are a helpful mental health assistant.
Below are some relevant FAQs. Use these as context to answer the user's question.

Context:
{context}

User's question: {user_question}

Answer:"""
    # Generate response
    response = generator(prompt, max_new_tokens=200, temperature=0.7, do_sample=True)
    # Only return the generated answer portion
    return response[0]['generated_text'][len(prompt):].strip()

# ------------------------------
# 6. Main Interactive Loop
# ------------------------------
if __name__ == "__main__":
    while True:
        print("\n🟢 Mental Health FAQ RAG (LOCAL) — Ask any question ('exit' to quit)")
        user_q = input("\nYou: ")
        if user_q.strip().lower() == "exit":
            print("Goodbye!")
            break
        results = retrieve(user_q, top_k=3)
        print("\nTop relevant FAQ answers:")
        for i, row in results.iterrows():
            print(f"\nQ: {row['Questions']}\nA: {row['Answers']}")
        print("\n🤖 RAG Answer (Local LLM):")
        print(rag_answer_local(user_q, results))
