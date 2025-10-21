package jnn.treinamento;

import java.util.concurrent.CountDownLatch;
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
	 * Operador multithread.
	 */
	ExecutorService exec;

	/**
	 * Quantidade de threads usadas pelo treinador.
	 */
	int threads;

	/**
	 * Clones do modelo base
	 * //TODO: utilizar outra abordagem que não envolva clonagem de modelos
	 */
	Modelo[] clones;

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
		if (super.numThreads == 1) {// config padrão
			threads = (int) (Runtime.getRuntime().availableProcessors() * 0.25) + 1;
		} else {
			threads = super.numThreads;
		}

		if (threads > x.length) threads = x.length;

        exec = Executors.newFixedThreadPool(threads);

		if (logs) esconderCursor();
		Variavel perdaEpoca = new Variavel();
		for (int e = 1; e <= epochs; e++) {
			embaralhar(x, y);
			perdaEpoca.zero();

			for (int i = 0; i < amostras; i += _tamLote) {
				int idFim = Math.min(i + _tamLote, amostras);
				Tensor[] loteX = utils.subArray(x, i, idFim);
				Tensor[] loteY = utils.subArray(y, i, idFim);

                modelo.gradZero();
                processoLote(loteX, loteY, loss, perdaEpoca);
                otm.update();      
			}

			if (logs) {
				limparLinha();
				exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca.get()/amostras));
			}

			// feedback de avanço
			if (calcHist) historico.add((perdaEpoca.get()/amostras));
		}

		if (logs) {
			exibirCursor();
			System.out.println();
		}

		exec.close();// tem que ter isso se não o processo do programa não acaba
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

		clones = new Modelo[threads];
		for (int j = 0; j < clones.length; j++) {
		    clones[j] = modelo.clone();
		}

        int blocoThread = Math.max(1, tamLote / threads);
        CountDownLatch latch = new CountDownLatch(threads);

        for (int t = 0; t < threads; t++) {
            final int id = t;
            final int inicio = t * blocoThread;
            final int fim = (t == threads - 1) ? tamLote : (t + 1) * blocoThread;

            exec.execute(() -> {
                try {
                    for (int j = inicio; j < fim; j++) {
                        Tensor prev = clones[id].forward(loteX[j]);
                        clones[id].backward(perda.backward(prev, loteY[j]));

						//feedback de avanço
                        if (calcHist) perdaEpoca.add(perda.forward(prev, loteY[j]).item());

						for (int c = 0; c < numCamadas; c++) {
							if (modelo.camada(c).treinavel()) {
								synchronized (modelo.camada(c)) {
                                    modelo.camada(c).gradKernel().add(clones[id].camada(c).gradKernel());
                                    modelo.camada(c).gradBias().add(clones[id].camada(c).gradBias());
                                }
                            }
                        }
                    }

				} catch (Exception e) {
					e.printStackTrace();
					System.exit(1);
                } finally {
                    latch.countDown();
                }
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
		
	}

}
