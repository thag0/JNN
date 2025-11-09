package jnn.avaliacao.metrica;

import jnn.core.tensor.Tensor;

/**
 *	F1 Score para avaliação de modelos de classificação.
 */
public class F1Score extends Metrica {

	/**
	 * Instancia a métrica do <strong>F1 Score</strong>.
	 * <p>
	 * 		O F1 Score é a média harmônica da precisão e da revocação, 
	 * 		considerando tanto a taxa de falsos positivos quanto a taxa 
	 * 		de falsos negativos. É particularmente útil quando se deseja 
	 * 		um equilíbrio entre precisão e revocação e quando há uma 
	 * 		distribuição desbalanceada entre as classes.
	 * </p>
	 */
	public F1Score() {}
	
	@Override
	public Tensor forward(Tensor[] entrada, Tensor[] real) {
		Tensor mat = super.matrizConfusao(entrada, real);
		double f1score = f1score(mat);

		return new Tensor(
			new double[]{ f1score }
		);
	}

	/**
	 * Calcula o valor f1 score.
	 * @param mat martiz de confusão
	 * @return f1 score.
	 */
	private double f1score(Tensor mat) {
		int nClasses = mat.shape()[0];

		double[] precisao = new double[nClasses];
		double[] recall = new double[nClasses];

		for (int i = 0; i < nClasses; i++) {
			int vp = (int) mat.get(i, i);// verdadeiro positivo
			int fp = 0;// falso positivo
			int fn = 0;// falso negativo


			for (int j = 0; j < nClasses; j++) {
				if (j != i) {
					fp += mat.get(j, i);
					fn += mat.get(i, j);
				}
			}

			// formulas da precisão e recall
			if ((vp + fp) > 0) precisao[i] = vp / (double)(vp + fp);
			else precisao[i] = 0.0;

			if ((vp + fn) > 0) recall[i] = vp / (double)(vp + fn);
			else recall[i] = 0.0;
		}

		double somaF1 = 0.0;
		for (int i = 0; i < nClasses; i++) {
			double f1Classe = 2.0 * (precisao[i] * recall[i]) / (precisao[i] + recall[i]);
			somaF1 += f1Classe;
		}

		return somaF1;
	}
}
