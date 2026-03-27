import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score
import os

def test_model_accuracy():
    # On vérifie si le modèle existe avant de tester
    if not os.path.exists("imdb_model.keras"):
        print("Le modèle n'est pas encore créé, test ignoré.")
        return 
    
    model = tf.keras.models.load_model("imdb_model.keras")
    
    # On charge des échantillons de test (que Hamad créera)
    if os.path.exists("test_samples.csv"):
        X_test = pd.read_csv("test_samples.csv")['review']
        y_test = pd.read_csv("test_labels.csv")['sentiment']
        
        predictions = (model.predict(X_test) >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Précision du modèle : {accuracy:.2f}")
        assert accuracy > 0.80, f"La précision {accuracy} est inférieure au seuil de 80%"
