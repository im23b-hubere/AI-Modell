import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. Lade die MedQuAD-Daten (jetzt vollständig)
with open('medquad_full.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)
print(f"Geladene QA-Paare: {len(faq_data)}")

# 2. Jede Frage ist eine eigene Kategorie (Intent)
questions = [item['question'] for item in faq_data]
answers = [item['answer'] for item in faq_data]
labels = [f"q_{i}" for i in range(len(questions))]  # Eindeutige Label-Strings

# 3. Bag-of-Words-Vektorisierung
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions).toarray()

# 4. Label-Encoding VOR dem Split
le = LabelEncoder()
labels_enc = le.fit_transform(labels)

# 5. Trainings- und Testdaten
X_train, X_test, y_train_enc, y_test_enc, answers_train, answers_test = train_test_split(
    X, labels_enc, answers, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train_enc = torch.tensor(y_train_enc, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test_enc = torch.tensor(y_test_enc, dtype=torch.long)

# 6. Einfaches Feedforward-Netz
class IntentNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_dim = X_train.shape[1]
hidden_dim = 32
output_dim = len(le.classes_)
model = IntentNet(input_dim, hidden_dim, output_dim)

# 7. Training vorbereiten
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 8. Training
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train_enc)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# 9. Testen
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test_enc).float().mean().item()
    print(f"\nTest accuracy: {accuracy*100:.2f}%")

# 10. Vorhersagefunktion

def predict_answer(user_question):
    bow = vectorizer.transform([user_question]).toarray()
    bow_tensor = torch.tensor(bow, dtype=torch.float32)
    with torch.no_grad():
        output = model(bow_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        # Hole die Antwort aus answers_train, falls im Training, sonst generische Antwort
        if pred_idx < len(answers_train):
            answer = answers_train[pred_idx]
        else:
            answer = "Sorry, I don't know the answer to that. Please consult a medical professional."
        confidence = torch.softmax(output, dim=1)[0, pred_idx].item()
    if confidence < 0.3:
        return "Sorry, I don't know the answer to that. Please consult a medical professional.", confidence
    return answer, confidence

if __name__ == "__main__":
    print("\nMedQuAD PyTorch Intent Chatbot (English, Full Dataset)")
    print("Type your medical question. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == 'exit':
            print("Bot: Take care! Always consult a real doctor for medical advice.")
            break
        answer, conf = predict_answer(user_input)
        print(f"Bot: {answer} (confidence: {conf:.2f})")
