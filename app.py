import sys
import os

# On ajoute le dossier 'src' au chemin de recherche de Python
# Cela permet de trouver 'app.py' qui est à l'intérieur
sys.path.append(os.path.abspath("src"))

# On importe l'interface Gradio depuis ton fichier src/app.py
from app import interface

# On lance l'application pour Hugging Face
if __name__ == "__main__":
    interface.launch()