import pandas as pd
import numpy as np
import joblib

# Charger le modèle sauvegardé (nouveau nom)
model = joblib.load("airbnb_model_xgb.pkl")

# Exemple de données à prédire (sans reviews)
new_data = pd.DataFrame([
    {
        "room_type": "Entire home/apt",
        "minimum_nights": 2,
        "availability_365": 180,
        "accommodates": 4,
        "bedrooms": 2,
        "bathrooms": 1,
        "beds": 3,
        "property_type": "Apartment",
        "neighbourhood_cleansed": "Downtown"
    },
    {
        "room_type": "Private room",
        "minimum_nights": 1,
        "availability_365": 90,
        "accommodates": 1,
        "bedrooms": 1,
        "bathrooms": 1,
        "beds": 1,
        "property_type": "House",
        "neighbourhood_cleansed": "Suburb"
    }
])

# Ajout des features calculées
new_data["bed_per_guest"] = new_data["beds"] / new_data["accommodates"]

# Valeurs moyennes de quartier (à adapter à ton historique réel)
mean_price_map = {
    "Downtown": 150,
    "Suburb": 90
}
new_data["mean_price_by_neighbourhood"] = new_data["neighbourhood_cleansed"].map(mean_price_map)

# Réorganiser les colonnes dans l'ordre attendu
expected_features = model.named_steps["preprocessing"].feature_names_in_
new_data = new_data[expected_features]

# Prédiction log(price)
log_pred = model.predict(new_data)

# Transformation inverse (prix)
price_pred = np.expm1(log_pred)

# Affichage
for i, price in enumerate(price_pred, 1):
    print(f"Logement {i} : Prix prédit = {price:.2f} €")
