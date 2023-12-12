import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
   caminho = './historico-perda.csv'
   dados = pd.read_csv(caminho)

   qtd_amostras = len(dados)
   eixo_y = dados.iloc[:, 0]
   eixo_x = list(range(qtd_amostras))

   plt.plot(eixo_x, eixo_y)
   plt.xlabel('Épocas')
   plt.ylabel('Perda')
   plt.title('Valor de perda em cada época')
   plt.show()
