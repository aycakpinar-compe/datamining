# Gerekli Kütüphaneler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Veri Yükleme ve Temizleme
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
data = pd.concat([train_data, test_data], axis=0)
data.dropna(inplace=True)
data.drop(columns=["Unnamed: 0", "id"], errors="ignore", inplace=True)

# Kategorik Değişken Dönüşümleri
label_encoder = LabelEncoder()
categorical_columns = ["Gender", "Customer Type", "Type of Travel", "satisfaction"]
data[categorical_columns] = data[categorical_columns].apply(label_encoder.fit_transform)
data["Class"] = data["Class"].map({"Eco": 0, "Eco Plus": 1, "Business": 2})

# Gecikme Sürelerini Toplama
data["Total_Delay"] = (
    data["Departure Delay in Minutes"] + data["Arrival Delay in Minutes"]
)
data.drop(
    columns=["Departure Delay in Minutes", "Arrival Delay in Minutes"], inplace=True
)

# Veri Bölme ve Normalizasyon
X = normalize(data.drop(columns=["satisfaction"]))
y = 1 - data["satisfaction"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Değerlendirme Fonksiyonu
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}\n{classification_report(y_test, y_pred)}")


# PCA ile Eğitim ve Tahmin
pca = PCA(n_components=1)  # Tek bileşene indiriyoruz
X_train_pca = pca.fit_transform(
    X_train
)  # PCA'yı eğitim verisine uygulayıp dönüştürüyoruz
X_test_pca = pca.transform(X_test)  # PCA ile test verisini dönüştürüyoruz

# PCA ile Modelleri Yeniden Eğitme ve Değerlendirme
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

for name, model in models.items():
    # Model eğitimi
    model.fit(X_train_pca, y_train)
    print(f"--- {name} ---")
    evaluate_model(model, X_test_pca, y_test)

    # Her modelin tahminlerini ayrı bir grafik olarak çizdirme
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test_pca)[:, 1]
        plt.figure(figsize=(8, 4))
        plt.plot(
            np.sort(X_test_pca.ravel()),
            np.sort(y_pred_prob),
            label=f"{name} Probability Curve",
            color="blue",
        )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Predicted Probability")
        plt.title(f"{name} - Predicted Probability Curve")
        plt.legend()
        plt.grid()
        plt.savefig(f"{name.lower().replace(' ', '_')}_probability_curve.png")
        plt.show()

# Üç modeli aynı grafik üzerinde karşılaştırma
plt.figure(figsize=(12, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test_pca)[:, 1]
        plt.plot(
            np.sort(X_test_pca.ravel()),
            np.sort(y_pred_prob),
            label=f"{name} Probability Curve",
        )
plt.xlabel("Principal Component 1")
plt.ylabel("Predicted Probability")
plt.title("Model Comparison - PCA")
plt.legend()
plt.grid()
plt.savefig("model_comparison.png")
plt.show()
