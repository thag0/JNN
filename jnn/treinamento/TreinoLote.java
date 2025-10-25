package jnn.treinamento;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.DoubleAdder;

import jnn.avaliacao.perda.Perda;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;
import jnn.otimizadores.Otimizador;

/**
  * Implementação de treino em lote dos modelos.
 */
public class TreinoLote extends MetodoTreino {
	
	/**
	 * Utilitário.
	 */
	Utils utils = new Utils();

	/**
	 * Operador multithread.
	 */
	ExecutorService exec;

	/**
	 * Tamanho do lote de amostras por iteração
	 */
	int tamLote;

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
	public TreinoLote(Modelo modelo, boolean hist, int tamLote) {
		super(modelo, hist);
		this.tamLote = tamLote;
	}

	public TreinoLote(Modelo modelo, int tamLote) {
		this(modelo, false, tamLote);
	}

	@Override
	protected void loop(Tensor[] x, Tensor[] y, Otimizador otm, Perda loss, int amostras, int epochs, boolean logs) {
		if (logs) esconderCursor();
		
		for (int e = 1; e <= epochs; e++) {
			embaralhar(x, y);
			DoubleAdder perdaEpoca = new DoubleAdder();

			for (int i = 0; i < amostras; i += tamLote) {
				int idFim = Math.min(i + tamLote, amostras);
				Tensor[] loteX = utils.subArray(x, i, idFim);
				Tensor[] loteY = utils.subArray(y, i, idFim);

                modelo.gradZero();
                processoLote(loteX, loteY, loss, perdaEpoca);
                otm.update();      
			}

			if (logs) {
				limparLinha();
				exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (perdaEpoca.sum()/amostras));
			}

			// feedback de avanço
			if (calcHist) historico.add((perdaEpoca.sum()/amostras));
		}

		if (logs) {
			exibirCursor();
			System.out.println();
		}

		exec.close();// tem que ter isso se não o processo do programa não acaba
	}

	/**
	 * Adapta a quantidade de threads usadas para prevenir overflow quando
	 * a quantidade de amostras do lote for diferente do configurado
	 * <p>
	 *		Essa assimetria geralmente ocorre no último lote do dataset,
	 *		quando a quantidade de amostras restante não é perfeitamente divisível
	 *		pelo tamanho do lote.
	 * </p>
	 * @param tamLote tamanho do lote da iteração.
	 */
	private void ajustarThreads(int tamLote) {
		int t = 0;

		if (_threads == 1) {// config padrão
			t = (int) (Runtime.getRuntime().availableProcessors() * 0.25) + 1;
		} else {
			t = _threads;
		}

		if (t > tamLote) t = tamLote;
		_threads = t;
	}

	/**
	 * Executa o passo de treino em lotes.
	 * @param loteX array de {@code Tensor} contendos entradas de treino.
	 * @param loteY array de {@code Tensor} contendos rótulos de treino.
	 * @param perda função de perda do modelo.
	 * @param perdaEpoca valor de perda por época de treinamento.
	 */
	private void processoLote(Tensor[] loteX, Tensor[] loteY, Perda perda, DoubleAdder perdaEpoca) {
		int tamLote = loteX.length;
		int numCamadas = modelo.numCamadas();

		ajustarThreads(tamLote);
		exec = Executors.newFixedThreadPool(_threads);

		clones = new Modelo[_threads];
		for (int j = 0; j < clones.length; j++) {
		    clones[j] = modelo.clone();
		}

        int blocoThread = Math.max(1, tamLote / _threads);
        CountDownLatch latch = new CountDownLatch(_threads);

        for (int t = 0; t < _threads; t++) {
            final int id = t;
            final int inicio = t * blocoThread;
            final int fim = (t == _threads - 1) ? tamLote : (t + 1) * blocoThread;

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
