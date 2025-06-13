import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("Prédiction du prix d'une annonce Airbnb")

# Charger le modèle
@st.cache_resource
def load_model():
    return joblib.load("airbnb_model_xgb_features.pkl")

model = load_model()
expected_features = list(model.named_steps["preprocessing"].feature_names_in_)

# Saisie utilisateur
st.header("Entrez les caractéristiques de votre annonce :")

room_type = st.selectbox("Type de logement", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
property_type = st.text_input("Type de propriété", "Apartment")
neighbourhood_cleansed = st.text_input("Quartier", "Downtown")
minimum_nights = st.number_input("Nuits minimum", min_value=1, max_value=30, value=2)
number_of_reviews = st.number_input("Nombre de reviews", min_value=0, value=10)
reviews_per_month = st.number_input("Reviews par mois", min_value=0.0, value=0.5, step=0.01)
availability_365 = st.number_input("Disponibilité (jours/an)", min_value=1, max_value=365, value=180)
accommodates = st.number_input("Capacité d'accueil", min_value=1, value=2)
bedrooms = st.number_input("Nombre de chambres", min_value=0, value=1)
bathrooms = st.number_input("Nombre de salles de bain", min_value=0.0, value=1.0, step=0.5)
beds = st.number_input("Nombre de lits", min_value=0, value=1)
mean_price_by_neighbourhood = st.number_input("Prix moyen du quartier (optionnel)", min_value=0.0, value=100.0, step=1.0)

if st.button("Prédire le prix"):
    # Calcul des features dérivées
    reviews_per_year = reviews_per_month * 12
    bed_per_guest = beds / accommodates if accommodates else 0
    price_per_accommodate = 0  # ignoré à la prédiction

    data = {
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "availability_365": availability_365,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "beds": beds,
        "property_type": property_type,
        "neighbourhood_cleansed": neighbourhood_cleansed,
        "reviews_per_year": reviews_per_year,
        "bed_per_guest": bed_per_guest,
        "price_per_accommodate": price_per_accommodate,
        "mean_price_by_neighbourhood": mean_price_by_neighbourhood
    }

    df = pd.DataFrame([data])
    df = df[expected_features]
    log_pred = model.predict(df)
    price_pred = float(np.expm1(log_pred)[0])
    st.success(f"Prix prédit : {price_pred:.2f} €")
