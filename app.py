import gradio as gr
import tensorflow as tf
import re

model = tf.keras.models.load_model("imdb_model.keras")

def clean_text_omega(text):
    text = str(text).lower()
    text = re.sub(r'<br />', ' ', text)
    # On garde les lettres et certains signes de ponctuation émotionnelle (! ?)
    text = re.sub(r'[^a-z!\?\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

    # --- DÉPLOIEMENT GRADIO ---
    def predict(text):
        res = model.predict([clean_text_ultima(text)], verbose=0)[0][0]
        label = "🌟 POSITIF" if res > 0.5 else "💀 NÉGATIF"
        score = res if res > 0.5 else 1 - res
        return f"{label}\nCertitude : {score:.2%}"

    gr.Interface(
        fn=predict, 
        inputs=gr.Textbox(label="Critique"), 
        outputs="text",
        title="🧠 IMDb SENTINEL ULTIMA",
        description="Architecture Transformer Hybrid-Pooling (Niveau Master 2 Recherche)."
    ).launch()

if __name__ == "__main__":
    interface.launch()