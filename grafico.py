import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
   caminho = './historico-perda.csv'
   dados = pd.read_csv(caminho)

   qtd_amostras = len(dados)
   eixo_y = dados.iloc[:, 0].tolist()
   eixo_x = list(range(qtd_amostras))
   min_y = min(eixo_y)

   plt.plot(eixo_x, eixo_y)
   plt.title(f'Perda por época (min = {min_y:.8f})')
   plt.xlabel('Épocas')
   plt.ylabel('Perda')
   plt.show()
