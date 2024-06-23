package jnn.treinamento;

import java.util.ArrayList;
import java.util.Random;

import jnn.avaliacao.perda.Perda;
import jnn.camadas.Camada;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;
import jnn.otimizadores.Otimizador;

/**
 * Em testes ainda.
 */
class TreinoLote {
	Utils utils = new Utils();
	AuxTreino aux = new AuxTreino();
	Random random = new Random();

	public boolean calcularHistorico = false;
	private ArrayList<Double> historico;
	boolean ultimoUsado = false;

	/**
	 * Implementação do treino em lote.
	 * @param historico
	 */
	public TreinoLote(boolean calcularHistorico) {
		historico = new ArrayList<>(0);
		this.calcularHistorico = calcularHistorico;
	}

	/**
	 * Configura a seed inicial do gerador de números aleatórios.
	 * @param seed nova seed.
	 */
	public void setSeed(long seed) {
		random.setSeed(seed);
		aux.setSeed(seed);
	}

	/**
	 * Configura o cálculo de custos da rede neural durante cada
	 * época de treinamento.
	 * @param calcular true armazena os valores de custo da rede, false não faz nada.
	 */
	public void setHistorico(boolean calcular) {
		calcularHistorico = calcular;
	}

	/**
	 * Treina o modelo por um número determinado de épocas usando o treinamento em lotes.
	 * @param modelo instância de modelo.
	 * @param entradas {@code Tensores} contendos os dados de entrada.
	 * @param saidas {@code Tensores} contendos os dados de saída (rótulos).
	 * @param epochs quantidade de épocas de treinamento.
	 * @param embaralhar embaralhar dados de treino para cada época.
	 * @param tamLote tamanho do lote.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void treinar(Modelo modelo, Tensor[] entradas, Tensor[] saidas, int epochs, int tamLote, boolean logs) {
		Camada[] camadas = modelo.camadas();
		Otimizador otimizador = modelo.otimizador();
		Perda perda = modelo.perda();
		int numAmostras = entradas.length;

		if (logs) aux.esconderCursor();
		double perdaEpoca;
		for (int e = 1; e <= epochs; e++) {
			aux.embaralharDados(entradas, saidas);
			perdaEpoca = 0;

			for (int i = 0; i < numAmostras; i += tamLote) {
				int fimIndice = Math.min(i + tamLote, numAmostras);
				Tensor[] entradaLote = aux.subArray(entradas, i, fimIndice);
				Tensor[] saidaLote = aux.subArray(saidas, i, fimIndice);
				
				modelo.zerarGrad();//zerar gradientes para o acumular pelo lote
				for (int j = 0; j < entradaLote.length; j++) {
					Tensor prev = modelo.forward(entradaLote[j]);

					if (calcularHistorico) {
						perdaEpoca += perda.calcular(prev, saidaLote[j]).item();
					}

					aux.backpropagation(camadas, perda, prev, saidaLote[j]);
				}

				otimizador.atualizar();          
			}

			if (logs) {
				aux.limparLinha();
				aux.exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca/numAmostras));
			}

			//feedback de avanço da rede
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
