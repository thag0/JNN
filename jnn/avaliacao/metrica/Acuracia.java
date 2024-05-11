package jnn.avaliacao.metrica;

import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

public class Acuracia extends Metrica {

	@Override
	public Tensor calcular(Modelo modelo, Tensor[] entrada, Tensor[] real) {
		Tensor[] prevs = modelo.forwards(entrada);
		int numAmostras = prevs.length;
		int acertos = 0;
		
		for (int i = 0; i < numAmostras; i++) {
			int idCalculado = indiceMaiorValor(prevs[i]);
			int idEsperado = indiceMaiorValor(real[i]);

			if(idCalculado == idEsperado) acertos++;
		}

		double acc = (double)acertos / numAmostras;
		return new Tensor(new double[]{ acc }, 1);
	}
}
