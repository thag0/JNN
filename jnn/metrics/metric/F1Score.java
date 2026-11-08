
package jnn.metrics.metric;

import jnn.core.tensor.Tensor;

/**
 *	F1 Score para avaliação de modelos de classificação.
 */
public class F1Score extends Metric {

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
	public Tensor forward(Tensor[] prev, Tensor[] real) {
		Tensor mat = super.confusionMatrix(prev, real);
		float f1score = f1score(mat);

		return new Tensor(1).set(f1score, 0);
	}

	/**
	 * Calcula o valor f1 score.
	 * @param mat martiz de confusão
	 * @return f1 score.
	 */
	private float f1score(Tensor mat) {
		int numClasses = mat.shape()[0];

		float[] precision = new float[numClasses];
		float[] recall    = new float[numClasses];

		for (int i = 0; i < numClasses; i++) {
			int vp = (int) mat.get(i, i);// verdadeiro positivo
			int fp = 0;// falso positivo
			int fn = 0;// falso negativo

			for (int j = 0; j < numClasses; j++) {
				if (j != i) {
					fp += mat.get(j, i);
					fn += mat.get(i, j);
				}
			}

			// formulas da precisão e recall
			if ((vp + fp) > 0) precision[i] = vp / (float)(vp + fp);
			else precision[i] = 0.0f;

			if ((vp + fn) > 0) recall[i] = vp / (float)(vp + fn);
			else recall[i] = 0.0f;
		}

		float f1Sum = 0.0f;
		for (int i = 0; i < numClasses; i++) {
			float f1Classe = 2.0f * (precision[i] * recall[i]) / (precision[i] + recall[i]);
			f1Sum += f1Classe;
		}

		return f1Sum;
	}
}
