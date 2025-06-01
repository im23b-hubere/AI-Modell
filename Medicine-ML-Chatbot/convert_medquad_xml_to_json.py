import os
import json
import xml.etree.ElementTree as ET

# Neuer Pfad zum MedQuAD-Repo (Unterordner MedQuAD)
MEDQUAD_DIR = os.path.join(os.path.dirname(__file__), 'medquad_raw', 'MedQuAD')
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), 'medquad_full.json')

qa_pairs = []

for root_dir, _, files in os.walk(MEDQUAD_DIR):
    for file in files:
        if file.endswith('.xml'):
            xml_path = os.path.join(root_dir, file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for qapair in root.findall('.//QAPair'):
                    q_text = qapair.findtext('Question')
                    a_text = qapair.findtext('Answer')
                    if q_text and a_text:
                        qa_pairs.append({
                            'question': q_text.strip(),
                            'answer': a_text.strip()
                        })
            except Exception as e:
                print(f"Fehler beim Parsen von {xml_path}: {e}")

print(f"Gefundene QA-Paare: {len(qa_pairs)}")

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

print(f"Fertig! Gespeichert als {OUTPUT_JSON}") 