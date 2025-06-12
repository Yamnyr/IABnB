import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from xgboost import XGBRegressor
from scipy.stats import uniform, randint
import shap
import matplotlib.pyplot as plt

# Chargement
df = pd.read_csv("listings.csv")
df["price"] = df["price"].replace(r'[\$,]', '', regex=True).astype(float)

# Nettoyage
features = [
    "room_type", "minimum_nights", "number_of_reviews", "reviews_per_month",
    "availability_365", "accommodates", "bedrooms", "bathrooms", "beds",
    "property_type", "neighbourhood_cleansed", "latitude", "longitude"
]
target = "price"

# Sélection + suppression des NaNs
df = df[features + [target]].dropna()
df = df[(df["price"] > 0) & (df["price"] < 1000)]
df = df[(df["minimum_nights"] <= 30) & (df["availability_365"] > 0)]

# Nouvelles features
center_lat, center_long = df["latitude"].mean(), df["longitude"].mean()
df["distance_to_center"] = np.sqrt((df["latitude"] - center_lat)**2 + (df["longitude"] - center_long)**2)
df["reviews_per_year"] = df["reviews_per_month"] * 12
df["bed_per_guest"] = df["beds"] / df["accommodates"]
df["price_per_accommodate"] = df["price"] / df["accommodates"]
df["mean_price_by_neighbourhood"] = df.groupby("neighbourhood_cleansed")["price"].transform("mean")

# Cible log-transformée
X = df.drop(columns=["price"])
y = np.log1p(df["price"])

# Prétraitement
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Pipeline XGBoost
xgb = XGBRegressor(random_state=42, verbosity=0)
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', xgb)
])

# Paramètres à optimiser
param_distributions = {
    'regressor__n_estimators': randint(100, 300),
    'regressor__max_depth': randint(3, 10),
    'regressor__learning_rate': uniform(0.01, 0.2),
    'regressor__subsample': uniform(0.7, 0.3),
    'regressor__colsample_bytree': uniform(0.7, 0.3)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    cv=3,
    scoring="neg_root_mean_squared_error",
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

# Évaluation
y_pred_log = best_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

r2_log = r2_score(y_test, y_pred_log)
r2_orig = r2_score(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
mae = mean_absolute_error(y_test_orig, y_pred)

print("\n🔍 Évaluation du modèle optimisé (XGBoost avec features avancées) :")
print(f"R² (log)     : {r2_log:.3f}")
print(f"R² (réel)    : {r2_orig:.3f}")
print(f"RMSE         : {rmse:.2f} €")
print(f"MAE          : {mae:.2f} €")

# Sauvegarde
joblib.dump(best_model, "airbnb_model_xgb_features.pkl")

# Test sur exemples
test_data = pd.DataFrame([
    {
        "room_type": "Entire home/apt",
        "minimum_nights": 3,
        "number_of_reviews": 120,
        "reviews_per_month": 1.5,
        "availability_365": 200,
        "accommodates": 4,
        "bedrooms": 2,
        "bathrooms": 1.5,
        "beds": 2,
        "property_type": "Apartment",
        "neighbourhood_cleansed": df["neighbourhood_cleansed"].iloc[0],
        "latitude": df["latitude"].iloc[0],
        "longitude": df["longitude"].iloc[0]
    },
    {
        "room_type": "Private room",
        "minimum_nights": 1,
        "number_of_reviews": 45,
        "reviews_per_month": 0.8,
        "availability_365": 150,
        "accommodates": 2,
        "bedrooms": 1,
        "bathrooms": 1,
        "beds": 1,
        "property_type": "House",
        "neighbourhood_cleansed": df["neighbourhood_cleansed"].iloc[1],
        "latitude": df["latitude"].iloc[1],
        "longitude": df["longitude"].iloc[1]
    }
])

# Recalcul des features sur test
for col in ["reviews_per_year", "bed_per_guest", "price_per_accommodate", "distance_to_center", "mean_price_by_neighbourhood"]:
    test_data[col] = 0  # valeur temporaire

test_data["reviews_per_year"] = test_data["reviews_per_month"] * 12
test_data["bed_per_guest"] = test_data["beds"] / test_data["accommodates"]
test_data["price_per_accommodate"] = 0  # ignoré à la prédiction
test_data["distance_to_center"] = np.sqrt((test_data["latitude"] - center_lat)**2 + (test_data["longitude"] - center_long)**2)
test_data["mean_price_by_neighbourhood"] = test_data["neighbourhood_cleansed"].map(
    df.groupby("neighbourhood_cleansed")["price"].mean()
)

# Réorganiser les colonnes
test_data = test_data[X.columns]
prices_pred_log = best_model.predict(test_data)
prices_pred = np.expm1(prices_pred_log)

print("\n💰 Prédictions sur exemples avec features avancées :")
for i, price in enumerate(prices_pred, 1):
    print(f"Logement {i} : {price:.2f} €")

# ========================
# 📊 Analyse SHAP
# ========================

print("\n📊 Analyse des variables avec SHAP...")

# Récupération du modèle XGBoost seul
booster = best_model.named_steps["regressor"]

# Transformation des données test avec le préprocesseur
X_test_transformed = best_model.named_steps["preprocessing"].transform(X_test)

# Création de l'explainer
explainer = shap.Explainer(booster)

# Calcul des valeurs SHAP
shap_values = explainer(X_test_transformed)

# Récupération des noms de features
feature_names = best_model.named_steps["preprocessing"].get_feature_names_out()

# Affichage du résumé
shap.summary_plot(shap_values, features=X_test_transformed, feature_names=feature_names)
