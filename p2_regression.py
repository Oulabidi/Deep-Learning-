import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# Daten laden
df = pd.read_csv("harvest.csv")
print("Check der ersten Zeilen:")
print(df.head())

# Features und Zielwert festlegen (Reihenfolge wichtig für Plot & Vorhersage)
X = df[["Dünger", "Niederschlag"]].values.astype(np.float32)
y = df["Ertrag"].values.astype(np.float32)

# Datensatz aufteilen (80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Netz-Struktur nach Vorgabe (2 Schichten à 4 Neuronen)
model = Sequential([
    layers.Dense(4, activation="relu", input_shape=(2,)),
    layers.Dense(4, activation="relu"),
    layers.Dense(1)
])

# Kompilieren mit Adam und MSE
model.compile(optimizer="adam", loss="mse")

# Training (100 Epochen)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=50,
    validation_data=(X_test, y_test),
    verbose=0
)

# Loss Plot (Model Loss)
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Test Loss")
plt.title("Training und Test Verlust")
plt.ylabel("Loss")
plt.xlabel("Epoche")
plt.legend()
plt.grid(True)
plt.show()

# 3D Visualisierung vorbereiten
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Datenpunkte einzeichnen
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c="blue", label="Trainingsdaten", alpha=0.6)
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c="green", label="Testdaten", alpha=0.6)

# Rote Vorhersagefläche generieren
d_range = np.linspace(0, 1, 30)
r_range = np.linspace(0, 1, 30)
D, R = np.meshgrid(d_range, r_range)
grid = np.column_stack([D.ravel(), R.ravel()])
Z = model.predict(grid, verbose=0).reshape(D.shape)

# Fläche plotten
surf = ax.plot_surface(D, R, Z, color="hotpink", alpha=0.3)

# Legende fixieren (Trick: Ein Dummy-Element für die Fläche hinzufügen)
import matplotlib.lines as mlines
pink_patch = mlines.Line2D([], [], color='hotpink', label='Vorhersage', linewidth=5)
blue_dots = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', label='Trainingsdaten')
green_dots = mlines.Line2D([], [], color='green', marker='o', linestyle='None', label='Testdaten')
ax.legend(handles=[blue_dots, green_dots, pink_patch])

ax.set_xlabel("Dünger")
ax.set_ylabel("Niederschlag")
ax.set_zlabel("Ertrag")
ax.set_title("3D Plot der Ertragsdaten und Modellvorhersagen")
plt.show()

# Finaler Test der Wertepaare
test_cases = np.array([[0.25, 0.25], [0.85, 0.75]], dtype=np.float32)
preds = model.predict(test_cases, verbose=0)

print("\n--- Vorhersage-Ergebnisse ---")
print(f"Dünger=0.25, Niederschlag=0.25 -> Ertrag: {preds[0][0]:.4f}")
print(f"Dünger=0.85, Niederschlag=0.75 -> Ertrag: {preds[1][0]:.4f}")