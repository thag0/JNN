package jnn.treinamento;

import java.util.LinkedList;

import jnn.avaliacao.perda.Perda;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;
import jnn.otimizadores.Otimizador;

/**
  * Implementação de treino em lote dos modelos.
 */
class TreinoLote {

	/**
	 * Auxiliar.
	 */
	AuxTreino aux = new AuxTreino();
	
	/**
	 * Utilitário.
	 */
	Utils utils = new Utils();

	/**
	 * Verificador para o calculo de perdas do modelo durante 
	 * o processo de treinamento.
	 */
	private boolean calcularHistorico = false;

	/**
	 * Histórico do modelo pelo processo de treinameto.
	 */
	private LinkedList<Double> historico;

	/**
	 * Verificador para rastrear o último modo de treino usado.
	 */
	boolean ultimoUsado = false;

	/**
	 * Implementação de treino em lote dos modelos.
	 * @param historico se {@code true} serão computados os valores de perda
	 * do modelo ao longo das épocas.
	 */
	public TreinoLote(boolean calcularHistorico) {
		historico = new LinkedList<>();
		setHistorico(calcularHistorico);
	}

	/**
	 * Configura a seed inicial do gerador de números aleatórios.
	 * @param seed nova seed.
	 */
	public void setSeed(Number seed) {
		aux.setSeed(seed);
	}

	/**
	 * Configura o cálculo dos valores de perda/custo do modelo
	 * durante o processo de treinamento.
	 * @param calcular se {@code true} serão computados os valores de 
	 * perda do modelo ao longo das épocas.
	 */
	public void setHistorico(boolean calcular) {
		calcularHistorico = calcular;
	}

	/**
	 * Treina o modelo por um número determinado de épocas usando o treinamento em lotes.
	 * @param modelo instância de modelo.
	 * @param x {@code Tensores} contendos os dados de entrada.
	 * @param y {@code Tensores} contendos os dados de saída (rótulos).
	 * @param epochs quantidade de épocas de treinamento.
	 * @param tamLote tamanho do lote.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void treinar(Modelo modelo, Tensor[] x, Tensor[] y, int epochs, int tamLote, boolean logs) {
		Otimizador otimizador = modelo.otimizador();
		Perda perda = modelo.perda();
		int numAmostras = x.length;

		if (logs) aux.esconderCursor();
		double perdaEpoca;
		for (int e = 1; e <= epochs; e++) {
			aux.embaralhar(x, y);
			perdaEpoca = 0;

			for (int i = 0; i < numAmostras; i += tamLote) {
				int fimId = Math.min(i + tamLote, numAmostras);
				Tensor[] loteX = utils.subArray(x, i, fimId);
				Tensor[] loteY = utils.subArray(y, i, fimId);
				
				modelo.zerarGrad();// zerar gradientes para o acumular pelo lote
				for (int j = 0; j < loteX.length; j++) {
					Tensor prev = modelo.forward(loteX[j]);

					if (calcularHistorico) {
						perdaEpoca += perda.calcular(prev, loteY[j]).item();
					}

					aux.backpropagation(modelo, prev, loteY[j]);
				}

				otimizador.atualizar();      
			}

			if (logs) {
				aux.limparLinha();
				aux.exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca/numAmostras));
			}

			// feedback de avanço do modelo
			if (calcularHistorico) historico.add((perdaEpoca/numAmostras));
		}

		if (logs) {
			aux.exibirCursor();
			System.out.println();
		}
	}

	/**
	 * Retorna o histórico de treino.
	 * @return histórico de treino.
	 */
	public Object[] historico() {
		return historico.toArray();
	}
}
