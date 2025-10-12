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
@SuppressWarnings("deprecation")// TODO: remover e adaptar para não usar Variavel
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
	protected void loop(Tensor[] x, Tensor[] y, Otimizador otm, Perda loss, int amostras, int epochs, boolean logs) {
		modelo.treino(true);

		if (logs) esconderCursor();
		Variavel perdaEpoca = new Variavel();
		for (int e = 1; e <= epochs; e++) {
			embaralhar(x, y);
			perdaEpoca.zero();

			for (int i = 0; i < amostras; i += _tamLote) {
				int fimId = Math.min(i + _tamLote, amostras);
				Tensor[] loteX = utils.subArray(x, i, fimId);
				Tensor[] loteY = utils.subArray(y, i, fimId);

                modelo.gradZero();
                processoLote(loteX, loteY, loss, perdaEpoca);
                otm.atualizar();      
			}

			if (logs) {
				limparLinha();
				exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca.get()/amostras));
			}

			// feedback de avanço
			if (calcularHistorico) historico.add((perdaEpoca.get()/amostras));
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
		
		int threads = numThreads;
		if (numThreads == 1) {
			threads = (int)(Runtime.getRuntime().availableProcessors() * 0.25);
		}
		if (threads > tamLote) threads = tamLote;

		Modelo[] clones = new Modelo[threads];
		for (int j = 0; j < clones.length; j++) {
		    clones[j] = modelo.clone();
		}

		int blocoThread = tamLote / threads;
		try (ExecutorService exec = Executors.newFixedThreadPool(threads)) {
		    for (int t = 0; t < threads; t++) {
		        final int id = t;
		        final int inicio = t * blocoThread;
		        final int fim = (t == threads - 1) ? tamLote : (t + 1) * blocoThread;

		        exec.execute(() -> {
		            for (int j = inicio; j < fim; j++) {
		                Tensor prev = clones[id].forward(loteX[j]);// forward no clone
						clones[id].backward(perda.backward(prev, loteY[j]));// backprop no clone
						
						if (calcularHistorico) {// feedback de avanço
							perdaEpoca.add(perda.forward(prev, loteY[j]).item());
						}
						
						// acumular gradientes no modelo original
						synchronized (modelo) {
							for (int c = 0; c < numCamadas; c++) {
								if (modelo.camada(c).treinavel()) {
									modelo.camada(c).gradKernel().add(clones[id].camada(c).gradKernel());
									modelo.camada(c).gradBias().add(clones[id].camada(c).gradBias());
								}
							}
		                }
		            }
		        });
		    }
		}
	}

}
