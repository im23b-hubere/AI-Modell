import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Lade die Intents-Daten (wir nutzen die gleiche Datei wie vorher)
with open('../Classic-ml-chatbot/intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# 2. Daten vorbereiten: Texte und Labels extrahieren
texts = []
labels = []
for intent in intents:
    for example in intent['examples']:
        texts.append(example)
        labels.append(intent['intent'])

# 3. Labels in Zahlen umwandeln
label2idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
idx2label = {idx: label for label, idx in label2idx.items()}
labels_idx = [label2idx[label] for label in labels]

# 4. Bag-of-Words-Vektorisierung
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
Y = np.array(labels_idx)

# 5. Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 6. In Torch-Tensoren umwandeln
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 7. Einfaches Feedforward-Netz definieren
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
output_dim = len(label2idx)
model = IntentNet(input_dim, hidden_dim, output_dim)

# 8. Training vorbereiten
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 9. Training
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# 10. Testen
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test).float().mean().item()
    print(f"\nTest-Genauigkeit: {accuracy*100:.2f}%")

# 11. Antwortlogik
antworten = {
    "greeting": "Hallo! Wie kann ich dir helfen?",
    "goodbye": "Tschüss! Bis zum nächsten Mal.",
    "how_are_you": "Mir geht es gut, danke der Nachfrage!",
    "thanks": "Gern geschehen!",
    "weather": "Ich bin leider kein Wetterfrosch, aber ich hoffe, das Wetter ist schön!",
    "name": "Ich bin dein kleiner PyTorch-Chatbot.",
    "age": "Ich bin so alt wie mein Code – also noch ziemlich jung!"
}

def intent_vorhersage(text):
    bow = vectorizer.transform([text]).toarray()
    bow_tensor = torch.tensor(bow, dtype=torch.float32)
    with torch.no_grad():
        output = model(bow_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        intent = idx2label[pred_idx]
    return intent

if __name__ == "__main__":
    print("\nPyTorch ML-Chatbot (Bag-of-Words)")
    print("(Tippe 'exit' zum Beenden)")
    while True:
        user_input = input("Du: ")
        if user_input.strip().lower() == 'exit':
            print("Bis bald!")
            break
        intent = intent_vorhersage(user_input)
        antwort = antworten.get(intent, "Das habe ich leider nicht verstanden.")
        print(f"Bot: {antwort} (Intent: {intent})") 