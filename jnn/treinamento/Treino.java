package jnn.treinamento;

import jnn.avaliacao.perda.Perda;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;
import jnn.otimizadores.Otimizador;

/**
  * Implementação de treino dos modelos.
 */
public class Treino extends Treinador {

	/**
	 * Treinador sequencial, atualiza os parâmetros a cada amostra de dados.
	 * @param historico modelo para treino.
	 */
	public Treino(Modelo modelo) {
		super(modelo);
	}
	
	@Override
	public void executar(Tensor[] xs, Tensor[] ys, int epochs, boolean logs) {
		modelo.treino(true);

		Otimizador otimizador = modelo.otimizador();
		Perda perda = modelo.perda();
		int numAmostras = xs.length;

		if (logs) esconderCursor();
		double perdaEpoca;
		for (int e = 1; e <= epochs; e++) {
			embaralhar(xs, ys);
			perdaEpoca = 0;
			
			for (int i = 0; i < numAmostras; i++) {
				Tensor prev = modelo.forward(xs[i]);
				
				//feedback de avanço
				if (calcularHistorico) {
					perdaEpoca += perda.calcular(prev, ys[i]).item();
				}
				
				modelo.gradZero();
				backpropagation(prev, ys[i]);
				otimizador.atualizar();
			}

			if (logs) {
				limparLinha();
				exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca/numAmostras));
			}

			// feedback de avanço
			if (calcularHistorico) historico.add(perdaEpoca/numAmostras);
		}

		if (logs) {
			exibirCursor();
			System.out.println();
		}

		modelo.treino(false);
	}

}
