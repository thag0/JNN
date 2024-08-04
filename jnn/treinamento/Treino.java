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
	public void executar(Tensor[] x, Tensor[] y, int epochs, boolean logs) {
		Otimizador otimizador = modelo.otimizador();
		Perda perda = modelo.perda();
		int numAmostras = x.length;

		if (logs) esconderCursor();
		double perdaEpoca;
		for (int e = 1; e <= epochs; e++) {
			embaralhar(x, y);
			perdaEpoca = 0;
			
			for (int i = 0; i < numAmostras; i++) {
				Tensor prev = modelo.forward(x[i]);
				
				//feedback de avanço
				if (calcularHistorico) {
					perdaEpoca += perda.calcular(prev, y[i]).item();
				}
				
				modelo.zerarGrad();
				backpropagation(prev, y[i]);
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
	}

}
