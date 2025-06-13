import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from xgboost import XGBRegressor, plot_importance
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement
df = pd.read_csv("listings.csv")
df["price"] = df["price"].replace(r'[\$,]', '', regex=True).astype(float)

# Mise à jour de la liste des features avec longitude et latitude
features = [
    "room_type", "minimum_nights", "availability_365",
    "accommodates", "bedrooms", "bathrooms", "beds",
    "property_type", "neighbourhood_cleansed",
    "longitude", "latitude"
]
target = "price"

# Nettoyage
df = df[features + [target]].dropna()
df = df[(df["price"] > 30) & (df["price"] < 1000)]
df = df[(df["minimum_nights"] <= 30) & (df["availability_365"] > 0)]

# Nouvelles features disponibles dès la création
df["bed_per_guest"] = df["beds"] / df["accommodates"]
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

# Pipeline avec XGBoost
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

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Recherche hyperparamètres
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

print("\n🔍 Évaluation du modèle optimisé (XGBoost) :")
print(f"R² (log)     : {r2_log:.3f}")
print(f"R² (réel)    : {r2_orig:.3f}")
print(f"RMSE         : {rmse:.2f} €")
print(f"MAE          : {mae:.2f} €")

# Sauvegarde du modèle
joblib.dump(best_model, "airbnb_model_xgb.pkl")

# Test prédiction - ajout longitude et latitude
test_data = pd.DataFrame([{
    "room_type": "Entire home/apt",
    "minimum_nights": 3,
    "availability_365": 200,
    "accommodates": 4,
    "bedrooms": 2,
    "bathrooms": 1.5,
    "beds": 2,
    "property_type": "Apartment",
    "neighbourhood_cleansed": df["neighbourhood_cleansed"].iloc[0],
    "longitude": df["longitude"].iloc[0],
    "latitude": df["latitude"].iloc[0],
}, {
    "room_type": "Private room",
    "minimum_nights": 1,
    "availability_365": 150,
    "accommodates": 2,
    "bedrooms": 1,
    "bathrooms": 1,
    "beds": 1,
    "property_type": "House",
    "neighbourhood_cleansed": df["neighbourhood_cleansed"].iloc[1],
    "longitude": df["longitude"].iloc[1],
    "latitude": df["latitude"].iloc[1],
}])

# Ajout des features calculées
test_data["bed_per_guest"] = test_data["beds"] / test_data["accommodates"]
test_data["mean_price_by_neighbourhood"] = test_data["neighbourhood_cleansed"].map(
    df.groupby("neighbourhood_cleansed")["price"].mean()
)

# Réorganiser les colonnes
test_data = test_data[X.columns]

prices_pred_log = best_model.predict(test_data)
prices_pred = np.expm1(prices_pred_log)

print("\n💰 Prédictions sur exemples :")
for i, price in enumerate(prices_pred, 1):
    print(f"Logement {i} : {price:.2f} €")

# Visualisations

# 1. Distribution des prix (réel et log)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(df['price'], bins=50, color='skyblue')
plt.title("Distribution des prix réels")
plt.xlabel("Prix (€)")
plt.ylabel("Nombre d'annonces")

plt.subplot(1,2,2)
plt.hist(np.log1p(df['price']), bins=50, color='salmon')
plt.title("Distribution des prix (log-transformés)")
plt.xlabel("log1p(Prix)")
plt.ylabel("Nombre d'annonces")

plt.tight_layout()
plt.show()

# 2. Prix moyen par quartier (top 10)
mean_price_neigh = df.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
mean_price_neigh.plot(kind='bar', color='purple')
plt.title("Top 10 quartiers par prix moyen")
plt.ylabel("Prix moyen (€)")
plt.xticks(rotation=45, ha='right')
plt.show()

# 3. Relation prix vs nombre de chambres (boxplot)
plt.figure(figsize=(10,6))
sns.boxplot(x='bedrooms', y='price', data=df[df['bedrooms'] <= 10])  # limiter outliers extrêmes
plt.title("Prix en fonction du nombre de chambres")
plt.ylabel("Prix (€)")
plt.xlabel("Nombre de chambres")
plt.show()

# 4. Corrélation entre variables numériques (heatmap)
plt.figure(figsize=(10,8))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation des variables numériques")
plt.show()

# 5. Importance des variables du modèle XGBoost
xgb_model = best_model.named_steps["regressor"]

plt.figure(figsize=(10,6))
plot_importance(xgb_model, max_num_features=10, importance_type='gain', height=0.8)
plt.title("Importance des 10 variables principales (Gain)")
plt.show()

# 6. Carte scatter longitude/latitude colorée par prix
plt.figure(figsize=(12,8))
plt.scatter(df['longitude'], df['latitude'], c=df['price'], cmap='viridis', alpha=0.4, s=10)
plt.colorbar(label='Prix (€)')
plt.title("Répartition spatiale des logements avec prix")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()