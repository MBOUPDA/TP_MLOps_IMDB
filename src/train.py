#Importation des bibliothèques
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#Fin de l'importation


#Définition des fonctions utilitaires

#Nettoyage du jeu de données
def clean(text):
    text = str(text).lower()
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'[^a-z!\?\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()
#Fin du nettoyage

#Chargement et préparation du jeu de données
def prepare_data(dataset):
    df = pd.read_csv(dataset, encoding='utf-8')
    df['review'] = df['review'].apply(clean)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)
#Fin du chargement et de la préparation

#Construction de l'architecture du modèle
def build_model(vocab_size=10000, sequence_length=256):
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=sequence_length)
    x = vectorizer(inputs)
    #Embedding dense
    x = tf.keras.layers.Embedding(vocab_size, 128)(x)
    x = tf.keras.layers.SpatialDropout1D(0.4)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(x) #Compréhension du flux séquentiel grace au GRU
    #Attention multi-tete
    attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.3)(x, x)
    x = tf.keras.layers.Add()([x, attention])
    x = tf.keras.layers.LayerNormalization()(x)
    #Global pooling contextuel
    avg_p = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_p = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([avg_p, max_p])
    #Bloc de décision
    x = tf.keras.layers.Dense(64, activation='swish')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    return model, vectorizer
#Fin de la construction

#Visualisation des résultats
def plot(history, save_path="results.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.legend()
    ax2.plot(history.history['accuracy'], label='Train')
    ax2.plot(history.history['val_accuracy'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.legend()
    plt.savefig(save_path)
    plt.close()
#Fin de la visualisation

#Fin de la définition


X_train, X_val, y_train, y_val = prepare_data("IMDB_dataset.csv")
model, vectorizer = build_model()
vectorizer.adapt(X_train.values)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)
]

#Entrainnement du modèle
print("\nDébut de l'entrainnement")
history = model.fit(
    X_train.values, y_train.values,
    validation_data=(X_val.values, y_val.values),
    epochs=1000,
    batch_size=64, 
    callbacks=callbacks
)
plot(history)
#Fin de l'entrainnement

#Sauvegarde    
model.save("best_feeling_classification_model.keras")
print("\nModèle 'best_feeling_classification_model.keras' bel et bien sauvegardé à la racine!")
#Fin de la sauvegarde  
