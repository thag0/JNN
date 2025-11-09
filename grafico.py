import matplotlib.pyplot as plt
import pandas as pd
import sys

if __name__ == '__main__':
   if len(sys.argv) < 2:
      print("Necessário infornar o nome do arquivo csv contendo os dados.")
      exit()

   caminho = sys.argv[1] + '.csv'
   dados = pd.read_csv(caminho)

   qtd_amostras = len(dados)
   eixo_y = dados.iloc[:, 0].tolist()
   eixo_x = list(range(qtd_amostras))
   min_y = min(eixo_y)

   fig, axes = plt.subplots(num="Train Loss")
   plt.plot(eixo_x, eixo_y, color='orange')
   plt.title(f'Perda por época (min = {min_y:.6f})')
   plt.xlabel('Épocas')
   plt.ylabel('Perda')
   plt.ylim(bottom=0)
   plt.grid()
   plt.show()
