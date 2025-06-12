import pandas as pd
import numpy as np
import joblib

# Charger le modèle sauvegardé
model = joblib.load("airbnb_model_xgb_features.pkl")

# Exemple de nouvelle donnée pour prédiction
new_data = pd.DataFrame([
    {
        "room_type": "Private room",
        "minimum_nights": 3,
        "number_of_reviews": 120,
        "reviews_per_month": 1.5,
        "availability_365": 200,
        "accommodates": 4,
        "bedrooms": 2,
        "bathrooms": 1.5,
        "beds": 2,
        "property_type": "Apartment",
        "neighbourhood_cleansed": "Downtown",  # adapte selon tes quartiers
        "reviews_per_year": 1.5 * 12,
        "bed_per_guest": 2 / 4,
        "price_per_accommodate": 0,  # ignoré à la prédiction
        "mean_price_by_neighbourhood": 130  # moyenne des prix pour "Downtown", adapte selon ton dataset
    }
])

# S’assurer que les colonnes sont dans le bon ordre
expected_features = model.named_steps["preprocessing"].feature_names_in_
new_data = new_data[expected_features]

# Prédiction (log prix)
log_pred = model.predict(new_data)

# Transformation inverse (prix réel)
price_pred = np.expm1(log_pred)

print(f"Prix prédit : {price_pred[0]:.2f} €")
