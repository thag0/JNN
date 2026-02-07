import matplotlib.pyplot as plt
import pandas as pd
import sys

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Necessário infornar o nome do arquivo csv contendo os dados.")
		exit()

	caminho = sys.argv[1]

	try:
		dados = pd.read_csv(caminho, header=None)
	except Exception as e:
		print(f"Erro ao ler arquivo {caminho}: {e}")
		exit()

	qtd_amostras = len(dados)
	qtd_colunas = dados.shape[1]

	if qtd_colunas == 1:
		eixo_y = dados.iloc[:, 0].tolist()
		eixo_x = list(range(qtd_amostras))
		min_y = min(eixo_y)

		fig, axes = plt.subplots(num="Treino")
		plt.plot(eixo_x, eixo_y, color='orange')
		plt.scatter(eixo_x, eixo_y, color='orange')
		plt.title(f'Perda por época (min = {min_y:.6f})')
		plt.xlabel('Épocas')
		plt.ylabel('Perda')
		plt.ylim(bottom=0)
		plt.grid()
		
	elif qtd_colunas == 2:
		eixo_y_loss = dados.iloc[:, 0].tolist()
		eixo_y_acc  = dados.iloc[:, 1].tolist()
		eixo_x = list(range(qtd_amostras))

		fig, (ax_loss, ax_acc) = plt.subplots(1, 2, num="Treino", figsize=(10, 4))

		ax_loss.plot(eixo_x, eixo_y_loss, color='orange')
		ax_loss.scatter(eixo_x, eixo_y_loss, color='orange')
		ax_loss.set_title('Perda por época')
		ax_loss.set_xlabel('Épocas')
		ax_loss.set_ylabel('Perda')
		ax_loss.set_ylim(bottom=0)
		ax_loss.grid()

		ax_acc.plot(eixo_x, eixo_y_acc, color='blue')
		ax_acc.scatter(eixo_x, eixo_y_acc, color='blue')
		ax_acc.set_title('Acurácia por época')
		ax_acc.set_xlabel('Épocas')
		ax_acc.set_ylabel('Acurácia')
		ax_acc.grid()

		plt.tight_layout()

	plt.show()