import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(
    page_title="Estimateur Prix Airbnb",
    page_icon="🏠",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("listings.csv")
    # Nettoyage rapide prix (enlever $ et ,)
    df["price"] = df["price"].str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    df["price"] = df["price"].astype(float)
    df = df.dropna(subset=["price", "longitude", "latitude"])
    return df

df = load_data()

# Navigation entre pages
page = st.sidebar.radio(
    "Navigation",
    ("Estimation", "Analyse des données")
)

if page == "Estimation":
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

    # ---------------------------
    # Formulaire d'estimation
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
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=2.3522, format="%.6f",
                                help="Longitude du logement (exemple: 2.3522 pour Paris)")
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=48.8566, format="%.6f",
                               help="Latitude du logement (exemple: 48.8566 pour Paris)")

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

    # Charger le modèle
    @st.cache_resource
    def load_model():
        return joblib.load("airbnb_model_xgb.pkl")

    model = load_model()
    expected_features = list(model.named_steps["preprocessing"].feature_names_in_)

    # Champs pour le nombre de reviews et notes
    number_of_reviews = st.number_input("Nombre de reviews", min_value=0, max_value=1000, value=20)
    review_scores_rating = st.number_input("Note moyenne des reviews", min_value=0.0, max_value=100.0, value=3.0)

    if st.button("💰 Estimer le prix de mon bien", type="primary"):
        if beds < accommodates * 0.3:
            st.warning("⚠️ Attention : Le nombre de lits semble faible par rapport à la capacité d'accueil")
        
        # Variables dérivées
        bed_per_guest = beds / accommodates if accommodates > 0 else 0
        price_per_accommodate = 0
        
        data = {
            "room_type": room_type,
            "minimum_nights": minimum_nights,
            "number_of_reviews": number_of_reviews,
            "availability_365": availability_365,
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "beds": beds,
            "property_type": property_type,
            "neighbourhood_cleansed": neighbourhood_cleansed,
            "bed_per_guest": bed_per_guest,
            "price_per_accommodate": price_per_accommodate,
            "mean_price_by_neighbourhood": mean_price_by_neighbourhood,
            "longitude": longitude,
            "latitude": latitude,
            "review_scores_rating": review_scores_rating
        }

        try:
            df_pred = pd.DataFrame([data])
            df_pred = df_pred[expected_features]
            log_pred = model.predict(df_pred)
            price_pred = float(np.expm1(log_pred)[0])
            
            st.success(f"💰 **Prix estimé : {price_pred:.0f} € par nuit**")
            
            monthly_potential = price_pred * availability_365 / 12
            yearly_potential = price_pred * availability_365
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prix par nuit", f"{price_pred:.0f} €")
            with col2:
                st.metric("Revenus mensuels potentiels", f"{monthly_potential:.0f} €")
            with col3:
                st.metric("Revenus annuels potentiels", f"{yearly_potential:.0f} €")
                
            st.info("💡 Ces revenus sont théoriques et basés sur votre disponibilité déclarée. "
                    "Le taux d'occupation réel dépend de nombreux facteurs (saisonnalité, concurrence, qualité de l'annonce, etc.)")
                    
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")
            st.info("Vérifiez que tous les champs sont correctement remplis.")
else:
    st.title("📊 Analyse exploratoire des données Airbnb")

    # 1. Distribution des prix (réel et log)
    st.subheader("Distribution des prix (réel et log-transformé)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df['price'], bins=50, color='skyblue')
    axes[0].set_title("Distribution des prix réels")
    axes[0].set_xlabel("Prix (€)")
    axes[0].set_ylabel("Nombre d'annonces")
    axes[1].hist(np.log1p(df['price']), bins=50, color='salmon')
    axes[1].set_title("Distribution des prix (log-transformés)")
    axes[1].set_xlabel("log1p(Prix)")
    axes[1].set_ylabel("Nombre d'annonces")
    plt.tight_layout()
    st.pyplot(fig)

    # 2. Prix moyen par quartier (top 10)
    st.subheader("Top 10 quartiers par prix moyen")
    mean_price_neigh = df.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    mean_price_neigh.plot(kind='bar', color='purple', ax=ax2)
    ax2.set_title("Top 10 quartiers par prix moyen")
    ax2.set_ylabel("Prix moyen (€)")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig2)

    # 3. Relation prix vs nombre de chambres (boxplot)
    st.subheader("Prix en fonction du nombre de chambres")
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.boxplot(x='bedrooms', y='price', data=df[df['bedrooms'] <= 10], ax=ax3)
    ax3.set_title("Prix en fonction du nombre de chambres")
    ax3.set_ylabel("Prix (€)")
    ax3.set_xlabel("Nombre de chambres")
    st.pyplot(fig3)

    # 4. Corrélation entre variables numériques (heatmap)
    st.subheader("Matrice de corrélation des variables numériques")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig4, ax4 = plt.subplots(figsize=(10,8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
    ax4.set_title("Matrice de corrélation des variables numériques")
    st.pyplot(fig4)

    # 5. Importance des variables du modèle XGBoost
    st.subheader("Importance des variables du modèle XGBoost")
    try:
        model = joblib.load("airbnb_model_xgb.pkl")
        xgb_model = model.named_steps["regressor"]
        from xgboost import plot_importance
        fig5, ax5 = plt.subplots(figsize=(10,6))
        plot_importance(xgb_model, max_num_features=10, importance_type='gain', height=0.8, ax=ax5)
        ax5.set_title("Importance des 10 variables principales (Gain)")
        st.pyplot(fig5)
    except Exception as e:
        st.info("Impossible d'afficher l'importance des variables : " + str(e))

    # 6. Carte scatter longitude/latitude colorée par prix
    st.subheader("Répartition spatiale des logements avec prix")
    fig6, ax6 = plt.subplots(figsize=(12,8))
    sc = ax6.scatter(df['longitude'], df['latitude'], c=df['price'], cmap='viridis', alpha=0.4, s=10)
    plt.colorbar(sc, ax=ax6, label='Prix (€)')
    ax6.set_title("Répartition spatiale des logements avec prix")
    ax6.set_xlabel("Longitude")
    ax6.set_ylabel("Latitude")
    st.pyplot(fig6)
