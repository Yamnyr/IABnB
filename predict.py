import pandas as pd
import numpy as np
import joblib

# Charger le modèle sauvegardé
model = joblib.load("airbnb_model_xgb_features.pkl")

# Exemple de données variées pour prédiction
new_data = pd.DataFrame([
    {
        "room_type": "Entire home/apt",
        "minimum_nights": 2,
        "number_of_reviews": 50,
        "reviews_per_month": 0.5,
        "availability_365": 180,
        "accommodates": 4,
        "bedrooms": 2,
        "bathrooms": 1,
        "beds": 3,
        "property_type": "Apartment",
        "neighbourhood_cleansed": "Downtown",
        "reviews_per_year": 0.5 * 12,
        "bed_per_guest": 3 / 4,
        "price_per_accommodate": 0,  # ignoré
        "mean_price_by_neighbourhood": 150  # adapter selon tes données
    },
    {
        "room_type": "Private room",
        "minimum_nights": 1,
        "number_of_reviews": 10,
        "reviews_per_month": 0.1,
        "availability_365": 90,
        "accommodates": 1,
        "bedrooms": 1,
        "bathrooms": 1,
        "beds": 1,
        "property_type": "House",
        "neighbourhood_cleansed": "Suburb",
        "reviews_per_year": 0.1 * 12,
        "bed_per_guest": 1 / 1,
        "price_per_accommodate": 0,
        "mean_price_by_neighbourhood": 90  # adapter selon tes données
    }
])

# S’assurer que les colonnes sont dans le bon ordre
expected_features = model.named_steps["preprocessing"].feature_names_in_
new_data = new_data[expected_features]

# Prédiction (log prix)
log_pred = model.predict(new_data)

# Transformation inverse (prix réel)
price_pred = np.expm1(log_pred)

for i, price in enumerate(price_pred, 1):
    print(f"Logement {i} : Prix prédit = {price:.2f} €")
