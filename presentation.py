import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger les modèles et encodeurs
crop_encoder = joblib.load('crops.pkl')
random_forest_model = joblib.load('random forest.pkl')
scaler=joblib.load('scaler.pkl')

# Créer les dictionnaires pour les classes des encodeurs
crop_class_dict = dict(enumerate(crop_encoder.classes_))

# Fonction pour styliser l'en-tête
def styled_header():
    st.markdown(
        """
        <style>
            /* Styles pour le titre */
            .title {
                font-size: 36px;
                font-weight: bold;
                color: #1E90FF;  /* Bleu */
            }
            /* Styles pour le texte en italique */
            .italic-text {
                font-style: italic;
                color: #808080;  /* Gris */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<p class='title'>Prédiction du rendement des cultures</p>", unsafe_allow_html=True)
    st.markdown("<p class='italic-text'>Optimisez vos récoltes avec l'analyse prédictive !</p>", unsafe_allow_html=True)
    st.write("")


# Fonction pour styliser la barre latérale
def styled_sidebar():
    st.sidebar.image('farm.jpg', use_column_width=True)
    st.sidebar.write("### Caractéristiques des cultures")
    crop_type = st.sidebar.selectbox("Type de culture", options=list(crop_class_dict.keys()), format_func=lambda x: crop_class_dict[x])
    Fertilizer = st.sidebar.number_input("Engrains (kg)")
    Rainfall = st.sidebar.number_input("Précipitations (mm)")
    st.sidebar.write("")
    return crop_type, Fertilizer, Rainfall

# Fonction pour afficher les prédictions
def display_predictions():
    crop_type, Fertilizer, Rainfall = styled_sidebar()
    if st.sidebar.button("Prédire"):
        x = np.array([crop_type, Rainfall, Fertilizer]).reshape(1, -1)
        x=scaler.transform(x)
        random_forest_prediction = random_forest_model.predict(x)[0]  # Récupérer le premier élément du tableau
        st.markdown(
            f"""
            <style>
                /* Styles pour la phrase */
                .prediction-label {{
                    color: #ffffff;  /* Blanc */
                }}
                /* Styles pour le résultat */
                .prediction-value {{
                    color: #00ff00;  /* Vert */
                    font-size: 48px; /* Taille de police agrandie */
                }}
            </style>
            <p class='prediction-label'>La prédiction du rendement de la culture est :</p>
            <p class='prediction-value'>{random_forest_prediction} T/ha</p>
            """,
            unsafe_allow_html=True
        )





# Fonction principale pour exécuter l'application
def main():
    styled_header()
    display_predictions()

# Exécution de l'application
if __name__ == "__main__":
    main()
