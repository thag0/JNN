package jnn.treinamento;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jnn.avaliacao.perda.Perda;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.Variavel;
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
		modelo.treino(true);
		
		Otimizador otimizador = modelo.otm();
		Perda perda = modelo.perda();
		int numAmostras = xs.length;

		if (logs) esconderCursor();
		Variavel perdaEpoca = new Variavel();
		for (int e = 1; e <= epochs; e++) {
			embaralhar(xs, ys);
			perdaEpoca.zero();

			for (int i = 0; i < numAmostras; i += _tamLote) {
				int fimId = Math.min(i + _tamLote, numAmostras);
				Tensor[] loteX = utils.subArray(xs, i, fimId);
				Tensor[] loteY = utils.subArray(ys, i, fimId);

                modelo.gradZero();
                processoLote(loteX, loteY, perda, perdaEpoca);
                otimizador.atualizar();      
			}

			if (logs) {
				limparLinha();
				exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca.get()/numAmostras));
			}

			// feedback de avanço
			if (calcularHistorico) historico.add((perdaEpoca.get()/numAmostras));
		}

		if (logs) {
			exibirCursor();
			System.out.println();
		}

		modelo.treino(false);
	}

	/**
	 * Executa o passo de treino em lotes.
	 * @param loteX array de {@code Tensor} contendos entradas de treino.
	 * @param loteY array de {@code Tensor} contendos rótulos de treino.
	 * @param perda função de perda do modelo.
	 * @param perdaEpoca valor de perda por época de treinamento.
	 */
	private void processoLote(Tensor[] loteX, Tensor[] loteY, Perda perda, Variavel perdaEpoca) {
		int tamLote = loteX.length;
		int numCamadas = modelo.numCamadas();

		int numThreads = Runtime.getRuntime().availableProcessors()/2;
		if (numThreads > tamLote) numThreads = tamLote;

		Modelo[] clones = new Modelo[numThreads];
		for (int j = 0; j < clones.length; j++) {
		    clones[j] = modelo.clone();
		}

		int blocoThread = tamLote / numThreads;
		try (ExecutorService exec = Executors.newFixedThreadPool(numThreads)) {
		    for (int t = 0; t < numThreads; t++) {
		        final int id = t;
		        final int inicio = t * blocoThread;
		        final int fim = (t == numThreads - 1) ? tamLote : (t + 1) * blocoThread;

		        exec.execute(() -> {
		            for (int j = inicio; j < fim; j++) {
		                Tensor prev = clones[id].forward(loteX[j]);

		                synchronized (modelo) {
							if (calcularHistorico) {// feedback de avanço
								perdaEpoca.add(perda.forward(prev, loteY[j]).item());
							}

							// copiar dados de cache dos clones e
							// parâmetros como somatório e máscara de dropout
		                    for (int c = 0; c < numCamadas; c++) {
		                        modelo.camada(c).copiarParaTreinoLote(clones[id].camada(c));
		                    }

		                    backpropagation(perda.backward(prev, loteY[j]));
		                }
		            }
		        });
		    }
		}
	}

}
