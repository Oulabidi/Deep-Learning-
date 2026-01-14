import pandas as pd
import matplotlib.pyplot as plt

# Daten laden
df = pd.read_csv("animals.csv")

# 1. Output: Die ersten 5 Zeilen in der Konsole
print("Erste 5 Zeilen des Datensatzes:")
print(df.head())

# 2. Output: Der Plot
plt.figure(figsize=(10, 7))

# Daten nach Tierart trennen für die Farben
moewe = df[df["Label"] == "Duennschnabelmoewe"]
schaf = df[df["Label"] == "Dickhornschaf"]

# Blau für Möwen, Rot für Schafe
plt.scatter(moewe["Groesse"], moewe["Umfang"], color="blue", label="Dünnschnabelmowe")
plt.scatter(schaf["Groesse"], schaf["Umfang"], color="red", label="Dickhornschaf")

# Beschriftungen
plt.title("Tiere nach Größe und Umfang")
plt.xlabel("Größe")
plt.ylabel("Umfang")

# Legende explizit nach rechts oben (loc='upper right')
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()