package jnn.treinamento;

import java.util.LinkedList;

import jnn.avaliacao.perda.Perda;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;
import jnn.otimizadores.Otimizador;

/**
  * Implementação de treino em lote dos modelos.
 */
class Treino {

	/**
	 * Auxiliar.
	 */
	AuxTreino aux = new AuxTreino();

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
	 * Implementação de treino sequencial dos modelos.
	 * @param historico se {@code true} serão computados os valores de perda
	 * do modelo ao longo das épocas.
	 */
	public Treino(boolean calcularHistorico) {
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
	 * Treino modelo por um número determinado de épocas.
	 * @param modelo modelo desejado.
	 * @param x {@code Tensores} contendos os dados de entrada.
	 * @param y {@code Tensores} contendos os dados de saída (rótulos).
	 * @param epochs quantidade de épocas de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void treinar(Modelo modelo, Tensor[] x, Tensor[] y, int epochs, boolean logs) {
		Otimizador otimizador = modelo.otimizador();
		Perda perda = modelo.perda();
		int numAmostras = x.length;

		if (logs) aux.esconderCursor();
		double perdaEpoca;
		for (int e = 1; e <= epochs; e++) {
			aux.embaralhar(x, y);
			perdaEpoca = 0;
			
			for (int i = 0; i < numAmostras; i++) {
				Tensor prev = modelo.forward(x[i]);
				
				//feedback de avanço da rede
				if (calcularHistorico) {
					perdaEpoca += perda.calcular(prev, y[i]).item();
				}
				
				modelo.zerarGrad();
				aux.backpropagation(modelo, prev, y[i]);
				otimizador.atualizar();
			}

			if (logs) {
				aux.limparLinha();
				aux.exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca/numAmostras));
			}

			// feedback de avanço da rede
			if (calcularHistorico) historico.add(perdaEpoca/numAmostras);
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
	public Object[] historico(){
		return historico.toArray();
	}
  
}
