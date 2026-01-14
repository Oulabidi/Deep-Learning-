import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Daten aus der CSV laden [cite: 5]
df = pd.read_csv("harvest.csv")

# Kurzer Check der ersten Zeilen [cite: 6]
print("Erste 5 Zeilen:")
print(df.head())

# Spalten für die Plots vorbereiten
duenger = df["Dünger"]
regen = df["Niederschlag"]
ertrag = df["Ertrag"]

# --- 3D-Visualisierung  ---
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# Rote Punkte wie in der Vorlage
ax.scatter(duenger, regen, ertrag, c='red', marker='o', alpha=0.6)

ax.set_title("3D-Plot der Ertragsdaten")
ax.set_xlabel("Dünger")
ax.set_ylabel("Niederschlag")
ax.set_zlabel("Ertrag")

plt.show()

# --- 2D-Vergleiche mit Seaborn  ---
# Erstellt zwei Plots nebeneinander
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Erster Plot: Dünger vs Ertrag
sns.scatterplot(data=df, x="Dünger", y="Ertrag", ax=ax1)
ax1.set_title("Dünger vs. Ertrag")

# Zweiter Plot: Niederschlag vs Ertrag
sns.scatterplot(data=df, x="Niederschlag", y="Ertrag", ax=ax2)
ax2.set_title("Niederschlag vs. Ertrag")

plt.tight_layout()
plt.show()