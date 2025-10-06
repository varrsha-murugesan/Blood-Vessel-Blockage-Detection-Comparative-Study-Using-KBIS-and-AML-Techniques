# ensemble_from_unet.py

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ----------------------------
# Load U-Net Model
# ----------------------------
unet_model = load_model("unet_model.h5", compile=False)

IMG_SIZE = 128

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=-1)

# ----------------------------
# Feature Extraction Function
# ----------------------------
def extract_features_from_mask(image):
    """Generate features from U-Net segmented output"""
    pred_mask = unet_model.predict(np.expand_dims(image, axis=0))[0, :, :, 0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8)

    vessel_area_ratio = np.sum(binary_mask) / binary_mask.size
    avg_intensity_vessel = np.mean(pred_mask[binary_mask == 1]) if np.sum(binary_mask) > 0 else 0
    perimeter = cv2.arcLength(cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True) if np.sum(binary_mask) > 0 else 0

    return [vessel_area_ratio, avg_intensity_vessel, perimeter]

# ----------------------------
# Load Dataset & Extract Features
# ----------------------------
image_dir = "dataset_split/test/images"
features, labels = [], []

# First pass – collect features only
for file in os.listdir(image_dir):
    img = load_image(os.path.join(image_dir, file))
    feats = extract_features_from_mask(img)
    features.append(feats)

# Convert to array for analysis
features = np.array(features)

# Compute adaptive thresholds from vessel_ratio distribution
ratios = features[:, 0]
low_thr = np.percentile(ratios, 33)
mid_thr = np.percentile(ratios, 66)

labels = []
for r in ratios:
    if r < low_thr:
        labels.append("Severe")
    elif r < mid_thr:
        labels.append("Moderate")
    else:
        labels.append("Normal")

print(f"Adaptive thresholds → Severe<{low_thr:.3f}, Moderate<{mid_thr:.3f}, Normal>={mid_thr:.3f}")


X = np.array(features)
y = np.array(labels)
print("Extracted features:", X.shape)
# ----------------------------
# Check Label Distribution
# ----------------------------
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

if len(unique) < 2:
    print("⚠️ Only one class found in dataset! Please adjust threshold values or use more varied images.")
    print("🛑 Stopping training to prevent single-class error.")
    exit()


# ----------------------------
# Encode & Split
# ----------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# ----------------------------
# Build Ensemble
# ----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(probability=True, kernel='rbf')
lr = LogisticRegression()

ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('svm', svm),
    ('lr', lr)
], voting='soft')

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

# ----------------------------
# Evaluate & Save
# ----------------------------
print("✅ Ensemble Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump(ensemble, "ensemble_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("💾 Ensemble model and label encoder saved successfully!")
