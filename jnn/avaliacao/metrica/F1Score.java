package jnn.avaliacao.metrica;

import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

public class F1Score extends Metrica{
	
	@Override
	public Tensor calcular(Modelo modelo, Tensor[] entrada, Tensor[] real) {
		Tensor mat = super.matrizConfusao(modelo, entrada, real);
		double f1score = f1score(mat);
		return new Tensor(new double[]{ f1score }, 1);
	}

	private double f1score(Tensor mat) {
		int nClasses = mat.shape()[0];

		double[] precisao = new double[nClasses];
		double[] recall = new double[nClasses];

		for (int i = 0; i < nClasses; i++) {
			int vp = (int)mat.get(i, i);//verdadeiro positivo
			int fp = 0;//falso positivo
			int fn = 0;//falso negativo


			for (int j = 0; j < nClasses; j++) {
				if (j != i) {
					fp += mat.get(j, i);
					fn += mat.get(i, j);
				}
			}

			//formulas da precisão e recall
			if ((vp + fp) > 0) precisao[i] = vp / (double)(vp + fp);
			else precisao[i] = 0;

			if ((vp + fn) > 0) recall[i] = vp / (double)(vp + fn);
			else recall[i] = 0;
		}

		double somaF1 = 0.0;
		for (int i = 0; i < nClasses; i++) {
			double f1Classe = 2 * (precisao[i] * recall[i]) / (precisao[i] + recall[i]);
			somaF1 += f1Classe;
		}

		return somaF1;
	}
}
