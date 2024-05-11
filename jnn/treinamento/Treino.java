package jnn.treinamento;

import java.util.ArrayList;
import java.util.Random;

import jnn.avaliacao.perda.Perda;
import jnn.camadas.Camada;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;
import jnn.otimizadores.Otimizador;

class Treino {
	AuxTreino aux = new AuxTreino();
	Utils utils = new Utils();
	Random random = new Random();

	private boolean calcularHistorico = false;
	private ArrayList<Double> historico;
	boolean ultimoUsado = false;

	/**
	 * Objeto de treino sequencial da rede.
	 * @param historico lista de custos da rede durante cada época de treino.
	 */
	public Treino(boolean calcularHistorico) {
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
	 * @param calcular caso verdadeiro, armazena os valores de custo da rede.
	 */
	public void setHistorico(boolean calcular) {
		calcularHistorico = calcular;
	}

	/**
	 * Treina a rede neural calculando os erros dos neuronios, seus gradientes para cada peso e 
	 * passando essas informações para o otimizador configurado ajustar os pesos.
	 * @param modelo instância da rede.
	 * @param entradas {@code Tensores} contendos os dados de entrada.
	 * @param saidas {@code Tensores} contendos os dados de saída (rótulos).
	 * @param epochs quantidade de épocas de treinamento.
	 * @param logs logs para perda durante as épocas de treinamento.
	 */
	public void treinar(Modelo modelo, Tensor[] entrada, Tensor[] saida, int epochs, boolean logs) {
		Camada[] camadas = modelo.camadas();
		Otimizador otimizador = modelo.otimizador();
		Perda perda = modelo.perda();
		int numAmostras = entrada.length;

		if (logs) aux.esconderCursor();
		double perdaEpoca;
		for (int e = 1; e <= epochs; e++) {
			aux.embaralharDados(entrada, saida);
			perdaEpoca = 0;
			
			for (int i = 0; i < numAmostras; i++) {
				Tensor prev = modelo.forward(entrada[i]);
				
				//feedback de avanço da rede
				if (calcularHistorico) {
					perdaEpoca += perda.calcular(prev, saida[i]).item();
				}
				
				modelo.zerarGradientes();
				aux.backpropagation(camadas, perda, prev, saida[i]);
				otimizador.atualizar(camadas);
			}

			if (logs) {
				aux.limparLinha();
				aux.exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca/numAmostras));
			}

			//feedback de avanço da rede
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
