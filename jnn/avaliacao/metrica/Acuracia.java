package jnn.avaliacao.metrica;

import jnn.core.tensor.Tensor4D;
import jnn.modelos.Modelo;

public class Acuracia extends Metrica{

	@Override
	public double calcular(Modelo modelo, Object entrada, Object[] saida) {
		if (!(saida instanceof double[][])) {
			throw new IllegalArgumentException(
				"Objeto esperado para saída é double[][], recebido " + saida.getClass().getTypeName()
			);
		}
			
		Object[] amostras = utils.transformarParaArray(entrada);
		Tensor4D[] prevs = modelo.forwards(amostras);
		int numAmostras = amostras.length;
		int acertos = 0;
		
		for (int i = 0; i < numAmostras; i++) {
			int idCalculado = indiceMaiorValor(prevs[i].paraArray());
			int idEsperado = indiceMaiorValor((double[])saida[i]);

			if(idCalculado == idEsperado) acertos++;
		}

		double acc = (double)acertos / numAmostras;
		return acc;
	}
}
