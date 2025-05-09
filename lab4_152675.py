import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymcdm.methods import TOPSIS, SPOTIS, VIKOR, PROMETHEE_II
from pymcdm.normalizations import minmax_normalization

# Definicja macierzy decyzyjnej i wag
alternatywy = ['Student A', 'Student B', 'Student C', 'Student W']
macierz_deczycyjna = np.array([
    [5, 3, 8, 6],     # Student A: umiarkowane spóźnienie, średni dramatyzm, dobra jakość, przyzwoity wkład
    [15, 7, 4, 7],    # Student B: dość duże spóźnienie, wysoki dramatyzm, przeciętna jakość, dobry wkład
    [10, 5, 6, 5],    # Student C: spóźniony, trochę dramatyzmu, przyzwoita jakość, średni wkład
    [30, 10, 5, 10]   # Student W: bardzo spóźniony, bardzo dramatyczny, średnia jakość, najlepszy wkład
])
wagi = np.array([0.4, 0.3, 0.2, 0.1])
typy_kryteriów = [-1, -1, 1, 1]  # Minimalizować spóźnienie i dramatyzm, maksymalizować jakość i wkład

# Normalizacja macierzy decyzyjnej
znormalizowana_macierz = minmax_normalization(macierz_deczycyjna, typy_kryteriów)

# Inicjalizacja metod MCDM
topsis = TOPSIS()
spotis = SPOTIS(np.stack((np.min(macierz_deczycyjna, axis=0), np.max(macierz_deczycyjna, axis=0)), axis=1))
vikor = VIKOR()
promethee = PROMETHEE_II('usual')

# Obliczanie wyników dla każdej metody
wyniki_topsis = topsis(znormalizowana_macierz, wagi, typy_kryteriów)
wyniki_spotis = spotis(macierz_deczycyjna, wagi, typy_kryteriów)
wyniki_vikor = vikor(macierz_deczycyjna, wagi, typy_kryteriów)
wyniki_promethee = promethee(macierz_deczycyjna, wagi, typy_kryteriów)

# Zebranie wyników w ramce danych
wyniki = pd.DataFrame({
    'Alternatywa': alternatywy,
    'TOPSIS': wyniki_topsis,
    'SPOTIS': wyniki_spotis,
    'VIKOR': wyniki_vikor,
    'PROMETHEE': wyniki_promethee
})
wyniki['Ranking TOPSIS'] = wyniki['TOPSIS'].rank(ascending=False)
wyniki['Ranking SPOTIS'] = wyniki['SPOTIS'].rank(ascending=True)
wyniki['Ranking VIKOR'] = wyniki['VIKOR'].rank(ascending=True)
wyniki['Ranking PROMETHEE'] = wyniki['PROMETHEE'].rank(ascending=False)

print('Wyniki analizy MCDM dla spóźnionych studentów:')
print(wyniki)

# Wizualizacja
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Wykres wyników
axs[0].bar(alternatywy, wyniki['TOPSIS'], label='TOPSIS', alpha=0.7)
axs[0].bar(alternatywy, wyniki['SPOTIS'], label='SPOTIS', alpha=0.7)
axs[0].bar(alternatywy, wyniki['VIKOR'], label='VIKOR', alpha=0.7)
axs[0].bar(alternatywy, wyniki['PROMETHEE'], label='PROMETHEE', alpha=0.7)
axs[0].set_title('Wyniki uzasadnienia spóźnionych prac')
axs[0].set_ylabel('Wynik')
axs[0].legend()

# Wykres porównań rankingów
axs[1].bar(alternatywy, wyniki['Ranking TOPSIS'], label='Ranking TOPSIS')
axs[1].bar(alternatywy, wyniki['Ranking SPOTIS'], label='Ranking SPOTIS')
axs[1].bar(alternatywy, wyniki['Ranking VIKOR'], label='Ranking VIKOR')
axs[1].bar(alternatywy, wyniki['Ranking PROMETHEE'], label='Ranking PROMETHEE')
axs[1].set_title('Porównanie rankingów studentów')
axs[1].set_ylabel('Ranking')
axs[1].invert_yaxis()
axs[1].legend()

plt.tight_layout()
plt.show()
