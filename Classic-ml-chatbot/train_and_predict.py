import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Lade die Trainingsdaten aus der JSON-Datei
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# 2. Bereite die Daten für das Training vor
texts = []  # Alle Beispielsätze
labels = []  # Zugehörige Intents
for intent in intents:
    for example in intent['examples']:
        texts.append(example)
        labels.append(intent['intent'])

# 3. Teile die Daten in Trainings- und Testdaten auf (zum Testen der Modellgüte)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 4. Erstelle eine Pipeline: Text-Vektorisierung + Klassifikator
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 5. Trainiere das Modell
model.fit(X_train, y_train)

# 6. Teste das Modell (optional, aber hilfreich zum Verständnis)
y_pred = model.predict(X_test)
print("\n--- Modellbewertung auf Testdaten ---")
print(classification_report(y_test, y_pred))

# 7. Funktion für Vorhersagen auf neue Benutzereingaben
def intent_vorhersage(text):
    return model.predict([text])[0]

# 8. Beispiel-Interaktion
if __name__ == "__main__":
    print("\nKleiner ML-Chatbot: Intent-Vorhersage")
    print("(Tippe 'exit' zum Beenden)")
    while True:
        user_input = input("Du: ")
        if user_input.strip().lower() == 'exit':
            print("Bis bald!")
            break
        intent = intent_vorhersage(user_input)
        print(f"(Erkannter Intent: {intent})") 