import pandas as pd
import numpy as np
import tensorflow as tf
import re
import gradio as gr
from sklearn.model_selection import train_test_split

# --- 1. PRÉTRAITEMENT ÉLITE ---
def clean_text_ultra(text):
    text = str(text).lower()
    text = re.sub(r'<br />', ' ', text)
    # On garde les marqueurs de sentiment : ! et ?
    text = re.sub(r'[^a-z!\?\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def prepare_data_ultra(csv_path):
    print("💎 Préparation du gisement de données IMDb...")
    df = pd.read_csv(csv_path, encoding='utf-8')
    df['review'] = df['review'].apply(clean_text_ultra)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# --- 2. L'ARCHITECTURE ULTRA-MAX (Stacked Gated-Hybrid) ---
def build_sentinel_ultra(vocab_size=10000, sequence_length=256):
    print("🏗️ Forge de l'architecture ULTRA-MAX (Bi-GRU + Attention)...")
    
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    x = vectorizer(inputs)
    
    # 1. Embedding dense avec Dropout
    x = tf.keras.layers.Embedding(vocab_size, 128)(x)
    x = tf.keras.layers.SpatialDropout1D(0.4)(x)
    
    # 2. La couche Gated (Bi-GRU) : Compréhension du flux séquentiel
    # Le GRU est plus rapide et souvent plus efficace que le LSTM pour le texte court
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(x)
    
    # 3. L'Attention Multi-Tête : Focus sur les termes clés après analyse du flux
    attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.3)(x, x)
    x = tf.keras.layers.Add()([x, attention])
    x = tf.keras.layers.LayerNormalization()(x)
    
    # 4. Global Pooling Contextuel
    avg_p = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_p = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([avg_p, max_p])
    
    # 5. Bloc de Décision "Deep-Dense"
    x = tf.keras.layers.Dense(64, activation='swish')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model, vectorizer

# --- 3. EXÉCUTION ---
def main():
    X_train, X_val, y_train, y_val = prepare_data_ultra("IMDB_dataset.csv")
    model, vectorizer = build_sentinel_ultra()
    
    print("🎯 Vectorisation du corpus...")
    vectorizer.adapt(X_train.values)
    
    # Stratégie de convergence rapide
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)
    ]
    
    print("🔥 Entraînement au paroxysme final...")
    model.fit(
        X_train.values, y_train.values,
        validation_data=(X_val.values, y_val.values),
        epochs=10,
        batch_size=64, # Augmenté pour la stabilité du gradient
        callbacks=callbacks
    )
    
    model.save("sentinel_ultra_max.keras")
    print("✅ Modèle ULTRA-MAX sauvegardé. Le sommet est atteint.")

    # UI Gradio
    def predict(text):
        p = model.predict([clean_text_ultra(text)], verbose=0)[0][0]
        label = "🌟 POSITIF" if p > 0.5 else "💀 NÉGATIF"
        return f"{label} (Score: {p:.2%})"

    gr.Interface(fn=predict, inputs="text", outputs="text", title="🧠 SENTINEL ULTRA-MAX").launch()

if __name__ == "__main__":
    main()