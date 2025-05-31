import re

# Einfache Datenstruktur für Regeln: Liste von (Pattern, Antwort)-Tupeln
rules = [
    (r"hallo|hi|hey", "Hallo! Wie kann ich dir helfen?"),
    (r"wie geht es dir", "Mir geht es gut, danke der Nachfrage! Und dir?"),
    (r"(wer|was) bist du", "Ich bin ein einfacher, regelbasierter Chatbot."),
    (r"hilfe", "Ich kann einfache Fragen beantworten. Probiere es aus!"),
    (r"(tschüss|bye|auf wiedersehen)", "Tschüss! Bis zum nächsten Mal."),
]

def antwort_finden(eingabe):
    """
    Sucht in der Benutzereingabe nach einem passenden Pattern und gibt die zugehörige Antwort zurück.
    Wenn keine Regel passt, wird eine Standardantwort gegeben.
    """
    eingabe = eingabe.lower()
    for pattern, antwort in rules:
        if re.search(pattern, eingabe):
            return antwort
    return "Das habe ich leider nicht verstanden. Kannst du das anders formulieren?"

if __name__ == "__main__":
    print("Willkommen beim einfachen regelbasierten Chatbot!")
    print("(Tippe 'bye' oder 'tschüss' zum Beenden)")
    while True:
        user_input = input("Du: ")
        if re.search(r"(bye|tschüss|auf wiedersehen)", user_input.lower()):
            print("Bot: Tschüss! Bis zum nächsten Mal.")
            break
        antwort = antwort_finden(user_input)
        print(f"Bot: {antwort}") 