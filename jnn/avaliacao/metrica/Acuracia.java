package jnn.avaliacao.metrica;

import jnn.modelos.Modelo;

public class Acuracia extends Metrica{

	@Override
	public double calcular(Modelo modelo, Object entrada, Object[] saida) {
		int acertos = 0;
		
		if (!(saida instanceof double[][])) {
			throw new IllegalArgumentException(
				"Objeto esperado para saída é double[][], recebido " + saida.getClass().getTypeName()
			);
		}
			
		Object[] arrEntrada = utils.transformarParaArray(entrada);
		int numAmostras = arrEntrada.length;
			
		for (int i = 0; i < numAmostras; i++) {
			modelo.forward(arrEntrada[i]);

			int indiceCalculado = super.indiceMaiorValor(modelo.saidaParaArray());
			int indiceEsperado = super.indiceMaiorValor((double[])saida[i]);

			if(indiceCalculado == indiceEsperado){
				acertos++;
			}
		}

		double acuracia = (double)acertos / numAmostras;
		return acuracia;
	}
}
