"""
train_models.py
Run this ONCE to train and save all ML models.
Command: python train_models.py
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────────────
# 1. STYLE CLASSIFIER (KNN)
# Features: [avg_brightness, contrast_std, avg_flow,
#            warm_ratio, cool_ratio, edge_density]
# Labels: Interview, Documentary, Travel, Cinematic, Action, Lifestyle
# ─────────────────────────────────────────────────────

print("Training style classifier...")

# Training data — each row is [brightness, contrast, motion, warmth, coolness, edges]
# These are realistic feature ranges for each style based on cinematography standards
style_X = np.array([
    # Interview / Portrait — bright, warm, stable, moderate edges
    [140, 45, 0.5, 0.7, 0.2, 0.3],
    [135, 42, 0.6, 0.65, 0.25, 0.28],
    [145, 48, 0.4, 0.72, 0.18, 0.32],
    [138, 44, 0.55, 0.68, 0.22, 0.29],
    [142, 46, 0.45, 0.71, 0.19, 0.31],
    [130, 40, 0.7, 0.60, 0.30, 0.25],
    [148, 50, 0.35, 0.75, 0.15, 0.35],

    # Documentary — neutral/cool, stable, high edges
    [110, 60, 1.2, 0.4, 0.55, 0.6],
    [105, 65, 1.5, 0.38, 0.58, 0.65],
    [115, 58, 1.0, 0.42, 0.52, 0.58],
    [108, 62, 1.3, 0.39, 0.56, 0.62],
    [112, 63, 1.1, 0.41, 0.54, 0.60],
    [100, 68, 1.8, 0.35, 0.60, 0.70],
    [118, 56, 0.9, 0.44, 0.50, 0.55],

    # Travel / Vlog — warm, handheld movement
    [150, 50, 4.5, 0.65, 0.28, 0.5],
    [155, 52, 5.0, 0.68, 0.25, 0.52],
    [148, 48, 4.2, 0.63, 0.30, 0.48],
    [158, 55, 5.5, 0.70, 0.22, 0.55],
    [145, 47, 4.0, 0.62, 0.32, 0.46],
    [160, 58, 6.0, 0.72, 0.20, 0.58],
    [152, 51, 4.8, 0.66, 0.27, 0.51],

    # Cinematic / Narrative — moody, controlled, high contrast
    [90,  75, 0.8, 0.45, 0.5, 0.7],
    [85,  80, 0.7, 0.42, 0.52, 0.72],
    [95,  72, 0.9, 0.48, 0.48, 0.68],
    [88,  78, 0.75, 0.43, 0.51, 0.71],
    [92,  76, 0.85, 0.46, 0.49, 0.69],
    [80,  85, 0.6, 0.40, 0.55, 0.75],
    [98,  70, 1.0, 0.50, 0.46, 0.66],

    # Action — shaky, high motion, high edges
    [120, 55, 9.0, 0.5, 0.4, 0.8],
    [115, 58, 10.0, 0.48, 0.42, 0.82],
    [125, 52, 8.5, 0.52, 0.38, 0.78],
    [118, 56, 9.5, 0.49, 0.41, 0.81],
    [122, 54, 9.2, 0.51, 0.39, 0.79],
    [110, 60, 11.0, 0.45, 0.45, 0.85],
    [128, 50, 8.0, 0.54, 0.36, 0.76],

    # Lifestyle / Commercial — very bright, clean, low edges
    [180, 35, 0.6, 0.6, 0.3, 0.2],
    [185, 32, 0.5, 0.62, 0.28, 0.18],
    [175, 38, 0.7, 0.58, 0.32, 0.22],
    [188, 30, 0.45, 0.64, 0.26, 0.16],
    [178, 36, 0.65, 0.59, 0.31, 0.21],
    [190, 28, 0.4, 0.66, 0.24, 0.14],
    [172, 40, 0.8, 0.56, 0.34, 0.24],
])

style_y = (
    ["Interview / Portrait"] * 7 +
    ["Documentary"] * 7 +
    ["Travel / Vlog"] * 7 +
    ["Cinematic / Narrative"] * 7 +
    ["Action"] * 7 +
    ["Lifestyle / Commercial"] * 7
)

style_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, metric="euclidean"))
])
style_pipeline.fit(style_X, style_y)
joblib.dump(style_pipeline, "models/style_classifier.pkl")
print("  Style classifier saved.")


# ─────────────────────────────────────────────────────
# 2. QUALITY SCORER (Random Forest)
# Features: [exposure_score, composition_score,
#            contrast_std, sharpness_var, stability_flow]
# Output: overall quality score 0-100
# ─────────────────────────────────────────────────────

print("Training quality scorer...")

np.random.seed(42)
n_samples = 500

# Generate realistic feature combos
exposure    = np.random.uniform(20, 100, n_samples)
composition = np.random.uniform(20, 100, n_samples)
contrast    = np.random.uniform(10, 90,  n_samples)
sharpness   = np.random.uniform(10, 500, n_samples)
stability   = np.random.uniform(0,  12,  n_samples)

quality_X = np.column_stack([exposure, composition, contrast, sharpness, stability])

# Quality score formula based on cinematography principles
# Exposure (25%) + Composition (25%) + Contrast (20%) + Sharpness (15%) + Stability (15%)
sharpness_normalized = np.clip(sharpness / 500 * 100, 0, 100)
stability_score = np.clip(100 - (stability * 8), 0, 100)

quality_y = (
    exposure    * 0.25 +
    composition * 0.25 +
    contrast    * 0.20 +
    sharpness_normalized * 0.15 +
    stability_score * 0.15
).astype(int)

quality_y = np.clip(quality_y, 0, 100)

quality_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])
quality_pipeline.fit(quality_X, quality_y)
joblib.dump(quality_pipeline, "models/quality_scorer.pkl")
print("  Quality scorer saved.")


# ─────────────────────────────────────────────────────
# 3. ENGAGEMENT PREDICTOR (Gradient Boosting)
# Features: [overall_score, style_encoded,
#            stability_score, exposure, composition]
# Output: Low / Medium / High
# ─────────────────────────────────────────────────────

print("Training engagement predictor...")

style_encoding = {
    "Lifestyle / Commercial": 6,
    "Cinematic / Narrative":  5,
    "Interview / Portrait":   4,
    "Travel / Vlog":          3,
    "Documentary":            2,
    "Action":                 1,
}

n_eng = 400
np.random.seed(99)

overall_scores  = np.random.uniform(20, 100, n_eng)
style_codes     = np.random.choice(list(style_encoding.values()), n_eng)
stab_scores     = np.random.uniform(20, 100, n_eng)
exp_scores      = np.random.uniform(20, 100, n_eng)
comp_scores     = np.random.uniform(20, 100, n_eng)

eng_X = np.column_stack([overall_scores, style_codes, stab_scores, exp_scores, comp_scores])

# Engagement label based on combined weighted score
eng_combined = (
    overall_scores * 0.35 +
    stab_scores    * 0.20 +
    exp_scores     * 0.20 +
    comp_scores    * 0.25
)

eng_y = np.where(eng_combined >= 70, "High",
        np.where(eng_combined >= 45, "Medium", "Low"))

eng_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42))
])
eng_pipeline.fit(eng_X, eng_y)
joblib.dump(eng_pipeline, "models/engagement_predictor.pkl")
print("  Engagement predictor saved.")


print("\n✅ All models trained and saved to /models folder!")
print("You only need to run this once.")