import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="EcoSort AI - Classification de Déchets",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #2E7D32;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DES CLASSES ---
CLASS_NAMES = [
    'Battery', 'Biological', 'Cardboard', 'Clothes', 'Glass', 
    'Metal', 'Paper', 'Plastic', 'Shoes', 'Trash'
]

# Dictionnaire de conseils de recyclage
RECYCLING_INFO = {
    'Battery': "🔋 Dangers chimiques. Ne pas jeter à la poubelle normale. Déposer en point de collecte.",
    'Biological': "🍎 Compostable. Mettre dans le bac à compost ou déchets organiques.",
    'Cardboard': "📦 Recyclable. A platir et mettre dans le bac de recyclage papier/carton.",
    'Clothes': "👕 Si bon état : donnez-les. Sinon, conteneur textile spécifique.",
    'Glass': "🍾 Recyclable à l'infini. A déposer dans les conteneurs à verre (sans bouchon).",
    'Metal': "🥫 Recyclable. Mettre dans le bac jaune (aluminium, conserves).",
    'Paper': "📄 Recyclable. Mettre dans le bac papier (éviter les papiers gras).",
    'Plastic': "🥤 Vérifier les consignes locales. Bouteilles et flacons vont généralement au tri.",
    'Shoes': "👞 Conteneur textile ou cordonnerie. Ne pas jeter dans la nature.",
    'Trash': "🗑️ Déchets non recyclables. A jeter dans la poubelle d'ordures ménagères."
}

# --- FONCTION DE CHARGEMENT DU MODÈLE (CACHÉE) ---
@st.cache_resource
def load_classification_model():
    model_path = 'best_model_EfficientNetB0.h5'
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

model = load_classification_model()

# --- FONCTION DE PRÉDICTION ---
def predict_image(image_data, model):
    # 1. Redimensionner l'image comme lors de l'entraînement (224, 224)
    size = (224, 224)
    
    # CORRECTION ICI : Utilisation de Image.LANCZOS au lieu de Image.ANTIALIAS
    try:
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
    except AttributeError:
        # Fallback si Image.LANCZOS n'est pas trouvé (très vieilles versions), on réessaie ANTIALIAS
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    
    # 2. Convertir en array numpy
    img_array = np.asarray(image)
    
    # 3. Gérer les images PNG avec transparence (4 canaux -> 3 canaux)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    # 4. Ajouter la dimension du batch (1, 224, 224, 3)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    # 5. Pré-traitement spécifique à EfficientNet
    preprocessed_img = preprocess_input(img_array_expanded)
    
    # 6. Prédiction
    prediction = model.predict(preprocessed_img)
    
    return prediction

# --- INTERFACE UTILISATEUR ---

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/recycle-sign.png", width=100)
    st.title("EcoSort AI")
    st.info("Ce modèle utilise l'architecture **EfficientNetB0** entraînée par Transfer Learning pour classer les déchets en 10 catégories.")
    st.markdown("---")
    st.write("📌 **Catégories supportées :**")
    st.markdown(", ".join(CLASS_NAMES))
    st.markdown("---")
    st.caption("Développé avec TensorFlow & Streamlit")

# Titre Principal
st.title("♻️ Assistant de Tri Intelligent")
st.markdown("Téléversez une image de déchet, et l'IA vous dira comment le trier !")

# Zone de chargement
uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Colonnes pour l'affichage (Image à gauche, Résultats à droite)
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### 📸 Votre Image")
        image = Image.open(uploaded_file)
        st.image(image, caption='Image téléversée', use_column_width=True)
    
    with col2:
        st.markdown("### 🔍 Analyse")
        
        if model is not None:
            with st.spinner('Analyse en cours...'):
                predictions = predict_image(image, model)
                
                # Récupérer la classe avec la plus haute probabilité
                predicted_class_index = np.argmax(predictions)
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                confidence = np.max(predictions) * 100
                
            # Affichage du résultat principal
            st.success(f"Résultat : **{predicted_class_name}**")
            
            # Barre de confiance
            st.write("Niveau de confiance :")
            st.progress(int(confidence))
            st.caption(f"{confidence:.2f}% de certitude")
            
            # Conseil de recyclage
            st.info(f"💡 **Conseil :** {RECYCLING_INFO[predicted_class_name]}")
            
            # Affichage détaillé des probabilités (Graphique)
            st.markdown("#### 📊 Détails des probabilités")
            
            chart_data = pd.DataFrame({
                'Catégorie': CLASS_NAMES,
                'Probabilité': predictions[0]
            })
            
            st.bar_chart(chart_data.set_index('Catégorie'))
            
        else:
            st.error("Le modèle n'a pas pu être chargé. Vérifiez que le fichier .h5 est présent.")

else:
    # Message d'accueil quand rien n'est chargé
    st.info("👆 Veuillez charger une image pour commencer l'analyse.")
