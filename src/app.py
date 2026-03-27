import gradio as gr
import tensorflow as tf
import numpy as np
import re
import random
from deep_translator import GoogleTranslator

# 1. LOGIQUE & MODÈLE
SAMPLES = [
    "Un pur chef-d'œuvre ! Les acteurs sont habités par leurs rôles et la mise en scène est sublime.",
    "Quelle claque visuelle ! La photographie est incroyable, chaque plan est un véritable tableau.",
    "Une comédie rafraîchissante avec un scénario original. À voir en famille sans hésiter.",
    "Absolument brillant ! Une pièce remarquablement écrite et jouée, une production magistrale.",
    "La fin est totalement inattendue ! Un thriller psychologique qui tient en haleine jusqu'au bout.",
    "Une performance magistrale de l'acteur principal qui mérite amplement une récompense.",
    "Une bande originale envoûtante qui porte littéralement le film. Une expérience sensorielle unique.",
    "Une réalisation audacieuse qui casse les codes du genre. Très impressionnant pour un premier film.",
    "Un hommage vibrant au cinéma classique. C'est nostalgique et techniquement irréprochable.",
    "L'alchimie entre les deux protagonistes est palpable, c'est le point fort de cette production.",
    "Ce film est une perte de temps totale. Le scénario est vide et la fin n'a aucun sens.",
    "Une catastrophe industrielle. Je n'ai jamais vu un montage aussi brouillon et des dialogues aussi plats.",
    "Vraiment déçu. Le rythme est lent, l'image est sombre et on s'ennuie ferme du début à la fin.",
    "Le jeu d'acteur laisse vraiment à désirer. On ne croit pas une seconde à cette intrigue.",
    "Encore un reboot inutile. Le cinéma manque cruellement d'inspiration ces derniers temps.",
    "Trop d'effets spéciaux tuent l'émotion. Le film ressemble plus à un jeu vidéo qu'à autre chose.",
    "Le scénario est rempli d'incohérences. Les personnages prennent des décisions absurdes.",
    "C'est visuellement beau mais terriblement creux. On s'ennuie malgré la beauté des décors.",
    "L'intrigue est prévisible dès la dixième minute. Aucune surprise, aucun frisson, un vrai navet.",
    "Le rythme est saccadé et l'histoire part dans tous les sens sans jamais vraiment conclure."
]

# Mots-clés pour valider le contexte (Cinéma/Critique)
CINEMA_KEYWORDS = [
    "film", "movie", "acteur", "actor", "scénario", "script", "histoire", "story", 
    "scène", "production", "cinéma", "réalisation", "director", "acting", "intrigue",
    "chef-d'oeuvre", "navet", "regardé", "vu", "watch", "filmé", "caméra"
]

try:
    model = tf.keras.models.load_model("best_feeling_classification_model.keras")
except:
    model = tf.keras.models.load_model("../models/best_feeling_classification_model.keras")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'[^a-z!\?\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict_sentiment(text):
    if not text or len(text.strip()) < 5:
        return "<div style='text-align:center; color:#888; padding:20px;'>📡 SCANNER EN ATTENTE...</div>", 0.5
    
    # --- VALIDATION DU CONTEXTE ---
    text_lower = text.lower()
    # On vérifie si au moins un mot-clé du cinéma est présent
    is_review = any(word in text_lower for word in CINEMA_KEYWORDS)
    
    if not is_review:
        # Message d'alerte Gradio (Popup orange en haut à droite)
        gr.Warning("⚠️ Détecté : Le texte ne semble pas être une critique de film.")
        return """
        <div style='text-align:center; padding:20px; border-radius:15px; background:rgba(255,165,0,0.1); border:1px solid #ffa500;'>
            <h3 style='color:#ffa500; font-family:Orbitron; margin:0;'>HORS CONTEXTE</h3>
            <p style='color:white; font-size:1em; margin-top:10px;'>Veuillez entrer une critique liée au cinéma pour une analyse précise.</p>
        </div>
        """, 0.5

    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        cleaned = clean_text(translated)
        
        if model:
            input_tensor = tf.convert_to_tensor([[cleaned]], dtype=tf.string)
            prediction = model.predict(input_tensor, verbose=0)[0][0]
        else:
            prediction = random.uniform(0.1, 0.9)
        
        color = "#00d4ff" if prediction > 0.5 else "#f5576c"
        label = "POSITIF" if prediction > 0.5 else "NÉGATIF"
        score = prediction if prediction > 0.5 else 1 - prediction

        return f"""
        <div style='text-align:center; padding:20px; border-radius:15px; background:rgba(255,255,255,0.05); border:1px solid {color}; box-shadow: 0 0 20px {color}44;'>
            <h2 style='color:{color}; font-family:Orbitron; margin:0; letter-spacing:2px;'>{label}</h2>
            <p style='color:white; font-size:1.2em; margin-top:10px;'>Confiance : {score:.2%}</p>
        </div>
        """, float(prediction)
    except Exception as e:
        return f"<div style='color:#ff4444;'>Erreur: {e}</div>", 0.5

# 2. CSS "ULTRA-WIDE & FULL-HEIGHT"
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Poppins:wght@300;400;600&display=swap');

body, html { margin: 0 !important; height: 100vh !important; background: #0f0c29; }
.gradio-container { max-width: 100% !important; margin: 0 !important; height: 100vh !important; background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important; }

.sidebar-panel {
    background: rgba(0, 0, 0, 0.2) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    height: 100vh !important;
    padding: 25px !important;
}

.glass-card {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 25px !important;
}

.title-text {
    font-family: 'Orbitron', sans-serif;
    background: linear-gradient(90deg, #00d4ff, #f5576c, #ffffff, #00d4ff);
    background-size: 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.2rem;
    font-weight: 700;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 5px;
    animation: gradientMove 8s linear infinite;
}
@keyframes gradientMove { 0% { background-position: 0% 50%; } 100% { background-position: 300% 50%; } }

.app-description {
    text-align: center;
    color: #ccc;
    font-family: 'Poppins', sans-serif;
    font-size: 1.1em;
    margin-bottom: 25px;
    letter-spacing: 1px;
}

footer { display: none !important; }
"""

# 3. INTERFACE GRADIO
with gr.Blocks(css=custom_css, title="Sentinel Omnibus-X") as interface:
    
    with gr.Row(equal_height=True):
        
        # COLONNE DE GAUCHE (SIDEBAR)
        with gr.Column(scale=1, elem_classes="sidebar-panel"):
            gr.HTML("""
                <div style='text-align: center; margin-bottom: 20px;'>
                    <img src="https://img.icons8.com/color/512/heart-with-pulse.png" width="90">
                    <h2 style='color:#00d4ff; font-family:Orbitron; margin: 10px 0 0 0; font-size:1.5em;'>SENTINEL v1.0</h2>
                    <p style='font-size: 0.8em; color: #888; letter-spacing:1px;'>GROUPE • 2 • M2-IABD</p>
                </div>
            """)
            gr.Markdown("---")
            gr.Markdown("\n\n🧪 **Status: `Active`**\n\n📡 **Engine: `TF-GRU`**")
            gr.HTML("<div style='flex-grow: 1;'></div>") 
            gr.Image("https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExNWE4eTZhczd1cWgxamI0dWdqZ2o2Y2l3eWRhMmNtbm4zOGVyNzh4YyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/QSzS0HpeTY5dwtgMEa/giphy.gif", show_label=False)
            gr.HTML("<p style='text-align:center; color:#444; font-size:0.7em; margin-top:10px;'>Master 2 AI & Big Data • 2026</p>")

        # COLONNE DE DROITE (CONTENU)
        with gr.Column(scale=4):
            gr.HTML('<h1 class="title-text">SENTINEL OMNIBUS-X</h1>')
            
            # --- AJOUT DE LA DESCRIPTION ---
            gr.HTML("""
                <div class="app-description">
                    Intelligence Artificielle spécialisée dans l'analyse de sentiment complexe pour les <b>critiques cinématographiques</b>
                </div>
            """)
            
            with gr.Tabs():
                with gr.Tab("🧬 DIAGNOSTIC EXPERT"):
                    with gr.Row(elem_classes="glass-card"):
                        with gr.Column(scale=2):
                            input_text = gr.Textbox(
                                label="🧠 FLUX PSYCHIQUE (Critique de film)", 
                                placeholder="Exemple: Ce film était incroyable, le jeu d'acteur est parfait...", 
                                lines=10,
                                elem_id="input-box"
                            )
                            btn_gen = gr.Button("🎲 GÉNÉRER UN TEST ALÉATOIRE", variant="primary")
                        
                        with gr.Column(scale=1):
                            res_html = gr.HTML("<div style='text-align:center; color:#888; padding-top:60px;'>PRÊT POUR ANALYSE...</div>")
                            slider = gr.Slider(0, 1, value=0.5, label="SPECTRE DE POLARITÉ", interactive=False)
                    
                    input_text.change(predict_sentiment, input_text, [res_html, slider])
                    btn_gen.click(lambda: random.choice(SAMPLES), None, input_text)

                with gr.Tab("📊 STATISTIQUES"):
                    with gr.Column(elem_classes="glass-card"):
                        gr.Markdown("### Architecture Hybride CNN-GRU")
                        gr.Label(label="Performance", value={"Précision Globale": 0.88, "F1-Score": 0.87})
                        gr.Markdown("""
                        **Note technique :** L'application utilise une validation par mots-clés pour s'assurer que l'entrée correspond au domaine d'entraînement (IMDb Dataset)!
                        """)

if __name__ == "__main__":
    interface.launch(show_api=False)
