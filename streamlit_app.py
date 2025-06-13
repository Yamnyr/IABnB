import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Estimateur Prix Airbnb",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Estimateur de Prix Airbnb")
st.markdown("**Obtenez une estimation du prix de votre bien pour la location courte durée**")

# Sidebar avec informations
with st.sidebar:
    st.markdown("### ℹ️ À propos")
    st.markdown("""
    Cet outil utilise un modèle d'intelligence artificielle 
    entraîné sur des milliers d'annonces Airbnb pour estimer 
    le prix optimal de votre bien.
    
    **Facteurs pris en compte :**
    - Type et taille du logement
    - Localisation
    - Capacité d'accueil
    - Conditions de location
    """)
    
    st.markdown("### 💡 Conseils")
    st.markdown("""
    - Soyez précis dans vos informations
    - Vérifiez les prix dans votre quartier
    - Adaptez selon la saisonnalité
    - Considérez vos charges et frais
    """)

# Charger le modèle
@st.cache_resource
def load_model():
    return joblib.load("airbnb_model_xgb_features.pkl")

model = load_model()
expected_features = list(model.named_steps["preprocessing"].feature_names_in_)

# Interface utilisateur organisée en sections
st.header("Estimez le prix de votre bien Airbnb")

# Section 1: Type de logement
st.subheader("🏠 Type de logement")
col1, col2 = st.columns(2)
with col1:
    room_type = st.selectbox("Type de location", 
                           ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
                           help="Louez-vous l'intégralité du logement ou seulement une chambre ?")
with col2:
    property_type = st.selectbox("Type de propriété", 
                               ["Apartment", "House", "Condominium", "Townhouse", "Loft", "Villa", "Studio", "Other"],
                               help="Quel type de bien proposez-vous ?")

# Section 2: Localisation
st.subheader("📍 Localisation")
neighbourhood_cleansed = st.text_input("Quartier", "Downtown", 
                                      help="Indiquez le quartier où se situe votre bien")

# Section 3: Capacité et équipements
st.subheader("🛏️ Capacité et équipements")
col1, col2, col3 = st.columns(3)
with col1:
    accommodates = st.number_input("Nombre de voyageurs", min_value=1, max_value=16, value=2,
                                 help="Combien de personnes peuvent loger ?")
with col2:
    bedrooms = st.number_input("Nombre de chambres", min_value=0, max_value=10, value=1,
                             help="Nombre de chambres séparées")
with col3:
    beds = st.number_input("Nombre de lits", min_value=1, max_value=20, value=1,
                         help="Nombre total de couchages")

bathrooms = st.number_input("Nombre de salles de bain", min_value=0.5, max_value=10.0, value=1.0, step=0.5,
                          help="Incluez les demi-salles de bain (0.5)")

# Section 4: Conditions de location
st.subheader("📋 Conditions de location")
col1, col2 = st.columns(2)
with col1:
    minimum_nights = st.number_input("Séjour minimum (nuits)", min_value=1, max_value=30, value=2,
                                   help="Durée minimale de séjour requise")
with col2:
    availability_365 = st.number_input("Disponibilité par an (jours)", min_value=30, max_value=365, value=250,
                                     help="Combien de jours par an comptez-vous louer ?")

# Section 5: Informations du marché local (optionnel)
st.subheader("💰 Informations du marché (optionnel)")
mean_price_by_neighbourhood = st.number_input("Prix moyen dans votre quartier (€)", 
                                            min_value=0.0, value=100.0, step=5.0,
                                            help="Si vous connaissez le prix moyen dans votre quartier")

if st.button("💰 Estimer le prix de mon bien", type="primary"):
    # Validation des données
    if beds < accommodates * 0.3:  # Vérification basique
        st.warning("⚠️ Attention : Le nombre de lits semble faible par rapport à la capacité d'accueil")
    
    # Calcul des features dérivées
    # Pour un nouveau bien, on initialise les reviews à 0
    number_of_reviews = 0
    reviews_per_month = 0.0
    reviews_per_year = 0.0
    bed_per_guest = beds / accommodates if accommodates > 0 else 0
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

    try:
        df = pd.DataFrame([data])
        df = df[expected_features]
        log_pred = model.predict(df)
        price_pred = float(np.expm1(log_pred)[0])
        
        # Affichage des résultats avec plus de détails
        # Affichage des résultats avec plus de détails
        st.success(f"💰 **Prix estimé : {price_pred:.0f} € par nuit**")

        # 🎈 Animation de ballons
        st.balloons()

        # Calculs additionnels pour donner plus de contexte
        monthly_potential = price_pred * availability_365 / 12
        yearly_potential = price_pred * availability_365
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prix par nuit", f"{price_pred:.0f} €")
        with col2:
            st.metric("Revenus mensuels potentiels", f"{monthly_potential:.0f} €")
        with col3:
            st.metric("Revenus annuels potentiels", f"{yearly_potential:.0f} €")
            
        st.info("💡 **Conseils :** Ces revenus sont théoriques et basés sur votre disponibilité déclarée. "
                "Le taux d'occupation réel dépend de nombreux facteurs (saisonnalité, concurrence, qualité de l'annonce, etc.)")
                
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {str(e)}")
        st.info("Vérifiez que tous les champs sont correctement remplis.")