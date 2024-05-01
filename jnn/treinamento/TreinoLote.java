package jnn.treinamento;

import java.util.ArrayList;
import java.util.Random;

import jnn.avaliacao.perda.Perda;
import jnn.camadas.Camada;
import jnn.core.OpArray;
import jnn.core.OpMatriz;
import jnn.core.Utils;
import jnn.modelos.Modelo;
import jnn.otimizadores.Otimizador;

/**
 * Em testes ainda.
 */
class TreinoLote {
	OpMatriz opmat = new OpMatriz();
	OpArray oparr = new OpArray();
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
	 * @param _perda função de perda (ou custo) usada para calcular os erros da rede.
	 * @param _otimizador otimizador configurado do modelo.
	 * @param entradas dados de entrada para o treino.
	 * @param saidas dados de saída correspondente as entradas para o treino.
	 * @param epochs quantidade de épocas de treinamento.
	 * @param embaralhar embaralhar dados de treino para cada época.
	 * @param tamLote tamanho do lote.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void treinar(Modelo modelo, Object entradas, Object[] saidas, int epochs, int tamLote, boolean logs) {
		Camada[] camadas = modelo.camadas();
		Otimizador otimizador = modelo.otimizador();
		Perda perda = modelo.perda();

		Object[] amostras = utils.transformarParaArray(entradas);
		Object[] rotulos = utils.transformarParaArray(saidas);
		int numAmostras = amostras.length;

		if (logs) aux.esconderCursor();
		double perdaEpoca;
		for (int e = 1; e <= epochs; e++) {
			aux.embaralharDados(amostras, rotulos);
			perdaEpoca = 0;

			for (int i = 0; i < numAmostras; i += tamLote) {
				int fimIndice = Math.min(i + tamLote, numAmostras);
				Object[] entradaLote = aux.obterSubMatriz(amostras, i, fimIndice);
				Object[] saidaLote = aux.obterSubMatriz(rotulos, i, fimIndice);
				
				modelo.zerarGradientes();//zerar gradientes para o acumular pelo lote
				for (int j = 0; j < entradaLote.length; j++) {
					double[] saidaAmostra = (double[]) saidaLote[j];
					modelo.forward(entradaLote[j]);

					if (calcularHistorico) {
						perdaEpoca += perda.calcular(modelo.saidaParaArray(), saidaAmostra);
					}

					aux.backpropagation(camadas, perda, modelo.saidaParaArray(), saidaAmostra);
				}

				otimizador.atualizar(camadas);          
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