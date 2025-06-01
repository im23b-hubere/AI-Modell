import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Lade die MedQuAD-Daten
with open('medquad_full.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)
print(f"Geladene QA-Paare: {len(faq_data)}")

questions = [item['question'] for item in faq_data]
answers = [item['answer'] for item in faq_data]

# 2. SentenceTransformer Modell laden
print("Lade Embedding-Modell (sentence-transformers)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Embeddings für alle Fragen berechnen
print("Berechne Embeddings für alle Fragen...")
question_embeddings = model.encode(questions, show_progress_bar=True, convert_to_numpy=True)

# 4. Retrieval-Funktion
def retrieve_answer(user_question, top_k=1):
    user_emb = model.encode([user_question], convert_to_numpy=True)
    sims = cosine_similarity(user_emb, question_embeddings)[0]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_question = questions[best_idx]
    answer = answers[best_idx]
    return answer, best_score, best_question

if __name__ == "__main__":
    print("\nMedQuAD Embedding Retrieval Chatbot (English, Full Dataset)")
    print("Type your medical question. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == 'exit':
            print("Bot: Take care! Always consult a real doctor for medical advice.")
            break
        answer, score, matched_q = retrieve_answer(user_input)
        print(f"Bot: {answer}\n(Matched: {matched_q}\nSimilarity: {score:.2f})\n") 