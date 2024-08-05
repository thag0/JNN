package jnn.treinamento;

import jnn.avaliacao.perda.Perda;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;
import jnn.otimizadores.Otimizador;

/**
  * Implementação de treino em lote dos modelos.
 */
public class TreinoLote extends Treinador {
	
	/**
	 * Utilitário.
	 */
	Utils utils = new Utils();

	/**
	 * Treinador em lotes, atualiza os parâmetros a subamostra
	 * do dataset fornecido.
	 * @param historico modelo para treino.
	 */
	public TreinoLote(Modelo modelo, int tamLote) {
		super(modelo, tamLote);
	}

	@Override
	public void executar(Tensor[] xs, Tensor[] ys, int epochs, boolean logs) {
		Otimizador otimizador = modelo.otimizador();
		Perda perda = modelo.perda();
		int numAmostras = xs.length;

		if (logs) esconderCursor();
		double perdaEpoca;
		for (int e = 1; e <= epochs; e++) {
			embaralhar(xs, ys);
			perdaEpoca = 0;

			for (int i = 0; i < numAmostras; i += tamLote) {
				int fimId = Math.min(i + tamLote, numAmostras);
				Tensor[] loteX = utils.subArray(xs, i, fimId);
				Tensor[] loteY = utils.subArray(ys, i, fimId);
				
				modelo.gradZero();// zerar gradientes para o acumular pelo lote
				for (int j = 0; j < loteX.length; j++) {
					Tensor prev = modelo.forward(loteX[j]);

					if (calcularHistorico) {// feedback de avanço
						perdaEpoca += perda.calcular(prev, loteY[j]).item();
					}

					backpropagation(prev, loteY[j]);
				}

				otimizador.atualizar();      
			}

			if (logs) {
				limparLinha();
				exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca/numAmostras));
			}

			// feedback de avanço
			if (calcularHistorico) historico.add((perdaEpoca/numAmostras));
		}

		if (logs) {
			exibirCursor();
			System.out.println();
		}
	}

}
