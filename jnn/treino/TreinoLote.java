package jnn.treino;

import java.util.concurrent.atomic.DoubleAdder;

import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.metrica.perda.Perda;
import jnn.modelos.Modelo;
import jnn.otm.Otimizador;

/**
  * Implementação de treino em lote dos modelos.
 */
public class TreinoLote extends MetodoTreino {
	
	/**
	 * Utilitário.
	 */
	Utils utils = new Utils();

	/**
	 * Tamanho do lote de amostras por iteração
	 */
	int tamLote;

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

			if (calcHist) {
				historico.add(perdaEpoca.sum() / amostras);
			}

			if (logs) {
				limparLinha();
				exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (perdaEpoca.sum()/amostras));
			}
			
		}

		if (logs) {
			exibirCursor();
			System.out.println();
		}
	}

	/**
	 * Executa o passo de treino em lotes.
	 * @param loteX array de {@code Tensor} contendos entradas de treino.
	 * @param loteY array de {@code Tensor} contendos rótulos de treino.
	 * @param loss função de perda do modelo.
	 * @param perdaEpoca valor de perda por época de treinamento.
	 */
	private void processoLote(Tensor[] loteX, Tensor[] loteY, Perda loss, DoubleAdder perdaEpoca) {
		Tensor real = utils.concatenar(loteY);

		Tensor prev = modelo.forward(utils.concatenar(loteX));
		Tensor g = loss.backward(prev, real);
		
		modelo.backward(g);

		if (calcHist) {
			double l = loss.forward(prev, real).item();
			int n = loteX.length;
			perdaEpoca.add(l * n);
		}
	}

}
