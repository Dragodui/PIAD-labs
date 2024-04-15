import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# zadanie 1

df1 = pd.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": ["a", "b", "a", "b", "b"],
})

grouped = df1.groupby("y")
mean_x = grouped["x"].mean()

print("Zadanie 1")
print(mean_x)

# zadanie 2

value_counts = df1["y"].value_counts()

print("Zadanie 2")
print(value_counts)

# zadanie 3

print("Zadanie 3")

df3_1 = np.loadtxt("autos.csv", skiprows=0, delimiter=",", dtype="str")
print(df3_1)

df3_2 = pd.read_csv("autos.csv", skiprows=0, index_col=0)
print(df3_2)

# zadanie 4

grouped4 = df3_2.groupby('make')

mean_fuel = grouped4["city-mpg"].mean()

print("Zadanie 4")
print(mean_fuel)

# zadanie 5

grouped_df = df3_2.groupby("make")
fuel_type_counts = grouped_df["fuel-type"].value_counts()

print("Zadanie 5")
print(fuel_type_counts)

# zadanie 6

x = df3_2["length"].to_numpy()
y = df3_2["city-mpg"].to_numpy()

coeffs1 = np.polyfit(x, y, 1)

coeffs2 = np.polyfit(x, y, 2)

y1 = np.polyval(coeffs1, x)
y2 = np.polyval(coeffs2, x)

print("Zadanie 6")
print(y1)
print(y2)

# zadanie 7

coeff7 = stats.pearsonr(x, y)[0]

print("Zadanie 7")
print(coeff7)

# zadanie 8

print("Zadanie 8")

plt.title("Zadanie 8")
plt.plot(x, y, "o", label="Próbki")
plt.plot(x, y1, label="Wielomian 1 stopnia")
plt.plot(x, y2, label="Wielomian 2 stopnia")

plt.xlabel("Długość")
plt.ylabel("Zużycie paliwa (miasto)")
plt.legend()
plt.show()

# zadanie 9

print("Zadanie 9")

kde = stats.gaussian_kde(x)
x_grid = np.linspace(min(x), max(x), 100)
y_kde = kde(x_grid)

plt.title("Zadanie 9")
plt.plot(x_grid, y_kde, label="Funkcja gęstości")
plt.plot(x, y, "o", label="Próbki")
plt.xlabel("Długość")
plt.ylabel("Gęstość prawdopodobieństwa")
plt.legend()
plt.show()

# zadanie 10

print("Zadanie 10")

x1 = df3_2["length"].to_numpy()
y1 = df3_2["city-mpg"].to_numpy()
x2 = df3_2["width"].to_numpy()
y2 = df3_2["city-mpg"].to_numpy()

plt.title("Zadanie 10")
plt.subplot(121)
plt.plot(x1, y1, "o", label="Próbki - Długość")
plt.xlabel("Długość")
plt.ylabel("Zużycie paliwa (miasto)")

plt.subplot(122)
plt.plot(x2, y2, "o", label="Próbki - Szerokość")
plt.xlabel("Szerokość")
plt.ylabel("Zużycie paliwa (miasto)")

plt.legend()
plt.show()

# zadanie 11

print("Zadanie 11")

x = df3_2["width"].to_numpy()
y = df3_2["length"].to_numpy()

kde = stats.gaussian_kde(np.vstack([x, y]))

x_grid = np.linspace(min(x), max(x), 100)
y_grid = np.linspace(min(y), max(y), 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
Z_grid = kde(np.vstack([X_grid, Y_grid]))

plt.figure(figsize=(8, 6))
plt.plot(x, y, "o", label="Próbki")
plt.contour(X_grid, Y_grid, Z_grid, cmap="viridis")
plt.xlabel("Szerokość")
plt.ylabel("Długość")
plt.legend()

plt.savefig("density_2d.png")
plt.savefig("density_2d.pdf")

plt.show()
