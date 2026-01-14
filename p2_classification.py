import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# 1. Daten laden
df = pd.read_csv("animals.csv")
print("Erste 5 Zeilen des Datensets:")
print(df.head())

# Labels kodieren: Moewe=0, Schaf=1
df['Label_num'] = df['Label'].map({'Duennschnabelmoewe': 0, 'Dickhornschaf': 1})

# Features festlegen (X=Größe, Y=Umfang)
X = df[["Groesse", "Umfang"]].values.astype(np.float32)
y = df["Label_num"].values.astype(np.float32)

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Modell-Struktur (2 Hidden Layers à 4 Neuronen)
def build_model():
    model = Sequential([
        layers.Dense(4, activation="relu", input_shape=(2,)),
        layers.Dense(4, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --- TEIL 1: Training mit wenigen Epochen (50) ---
model_few = build_model()
history_few = model_few.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_test, y_test), verbose=0)

plt.figure(figsize=(6, 4))
plt.plot(history_few.history["loss"], label="Training Loss")
plt.plot(history_few.history["val_loss"], label="Validation Loss")
plt.title("Model Loss (wenige Epochen)")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

# --- TEIL 2: Training mit vielen Epochen (250) ---
model_many = build_model()
history_many = model_many.fit(X_train, y_train, epochs=250, batch_size=100, validation_data=(X_test, y_test), verbose=0)

plt.figure(figsize=(6, 4))
plt.plot(history_many.history["loss"], label="Training Loss")
plt.plot(history_many.history["val_loss"], label="Validation Loss")
plt.title("Model Loss (viele Epochen)")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

# --- TEIL 3: Klassengebiet Plot ---

plt.figure(figsize=(10, 7))

# Grid für die Fläche
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Vorhersage für die Fläche (Modell mit vielen Epochen)
Z = model_many.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
Z = Z.reshape(xx.shape)

# Gelerntes Modell plotten (contourf)
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="bwr", alpha=0.3)

# Test-Daten einzeichnen (scatter)
plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c="blue", label="Test Dünnschnabelmowe")
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c="red", label="Test Dickhornschaf")

plt.title("Gelerntes Klassengebiet")
plt.xlabel("Größe")
plt.ylabel("Umfang")
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# --- TEIL 4: Vorhersage-Ergebnisse ---
test_samples = np.array([[90, 90], [70, 70]], dtype=np.float32)
preds = model_many.predict(test_samples, verbose=0)

print("\n--- Vorhersage-Ergebnisse ---")
for val, prob in zip(test_samples, preds):
    tier = "Dickhornschaf" if prob[0] > 0.5 else "Duennschnabelmoewe"
    print(f"Eingabe: Umfang {val[1]}, Größe {val[0]} -> Vorhersage: {tier} (Wahrsch. Schaf: {prob[0]:.4f})")