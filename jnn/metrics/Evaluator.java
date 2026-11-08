package jnn.metrics;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.metrics.loss.*;
import jnn.metrics.metric.*;
import jnn.models.Model;

/**
 * Contém implementações de métricas para analisar modelos.
 */
public class Evaluator {

	/**
	 * Modelo base de avaliação.
	 */
	private Model model;

	MSE mse = new MSE();
	MAE mae = new MAE();
	MSLE msle = new MSLE();
	RMSE rmse = new RMSE();
	CossEntropy ce = new CossEntropy();
	BinaryCrossEntropy bce = new BinaryCrossEntropy();
	
	ConfusionMatrix mc = new ConfusionMatrix();
	Accuracy acuracia = new Accuracy();
	F1Score f1Score = new F1Score();

	/**
	 * Instancia um novo avaliador destinado a um Modelo.
	 * @param model modelo base.
	 */
	public Evaluator(Model model) {
		this.model = JNNutils.validarNaoNulo(model, "model == null");
	}

	/**
	 * Calcula o erro médio quadrado em relação aos dados previstos e reais.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor mse(Tensor prev, Tensor real) {
		return mse.forward(prev, real);
	}

	/**
	 * Calcula o erro médio quadrado em relação aos dados previstos e reais.
	 * @param prev {@code Tensores} com dados previstos.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor mse(Tensor[] prev, Tensor[] real) {
		return mse(
			JNNutils.concatenar(prev),
			JNNutils.concatenar(real)
		);
	}

	/**
	 * Calcula o erro médio quadrado logarítimico em relação aos dados 
	 * previstos e reais.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor msle(Tensor prev, Tensor real) {
		return msle.forward(prev, real);
	}

	/**
	 * Calcula o erro médio quadrado logarítimico em relação aos dados 
	 * previstos e reais.
	 * @param prev {@code Tensores} com dados previstos.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor msle(Tensor[] prev, Tensor[] real) {
		return msle(
			JNNutils.concatenar(prev),
			JNNutils.concatenar(real)
		);
	}

	/**
	 * Calcula o erro médio absoluto em relação aos dados previstos e reais.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor mae(Tensor prev, Tensor real) {
		return mae.forward(prev, real);
	}

	/**
	 * Calcula o erro médio absoluto em relação aos dados previstos e reais.
	 * @param prev {@code Tensores} com dados previstos.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor mae(Tensor[] prev, Tensor[] real) {
		return mae(
			JNNutils.concatenar(prev),
			JNNutils.concatenar(real)
		);
	}

	/**
	 * Calcula a raiz do erro médio quadrado em relação aos dados previstos e reais.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor rmse(Tensor prev, Tensor real) {
		return rmse.forward(prev, real);
	}

	/**
	 * Calcula a raiz do erro médio quadrado em relação aos dados previstos e reais.
	 * @param prev {@code Tensores} com dados previstos.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor rmse(Tensor[] prev, Tensor[] real) {
		return rmse(
			JNNutils.concatenar(prev),
			JNNutils.concatenar(real)
		);
	}

	/**
	 * Calcula a entropia cruzada em relação aos dados previstos e reais.
	 * e as saídas reais fornecidas.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor crossEntropy(Tensor prev, Tensor real) {  
		return ce.forward(prev, real);
	}

	/**
	 * Calcula a entropia cruzada em relação aos dados previstos e reais.
	 * e as saídas reais fornecidas.
	 * @param prev {@code Tensores} com dados previstos.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor crossEntropy(Tensor[] prev, Tensor[] real) {  
		return crossEntropy(
			JNNutils.concatenar(prev),
			JNNutils.concatenar(real)
		);
	}

	/**
	 * Calcula a entropia cruzada binária entre as saídas previstas pela rede neural
	 * e as saídas reais fornecidas.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor binaryCrossEntropy(Tensor prev, Tensor real) {
		return bce.forward(prev, real);
	}

	/**
	 * Calcula a entropia cruzada binária entre as saídas previstas pela rede neural
	 * e as saídas reais fornecidas.
	 * @param prev {@code Tensores} com dados previstos.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor binaryCrossEntropy(Tensor[] prev, Tensor[] real) {
		return binaryCrossEntropy(
			JNNutils.concatenar(prev),
			JNNutils.concatenar(real)
		);
	}

	/**
	 * Calcula a precisão em relação aos dados de entrada e saída fornecidos.
	 * @param xs {@code Tensores} com dados de entrada para o modelo.
	 * @param ys {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor accuracy(Tensor[] xs, Tensor[] ys) {
		final int n = xs.length;
		int batch = 32;
		
		float hits = 0;
		for (int i = 0; i < n; i += batch) {
			int start = i;
			int end = Math.min(start + batch, n);
			Tensor[] x = JNNutils.subArray(xs, start, end);
			Tensor[] y = JNNutils.subArray(ys, start, end);
			Tensor[] prev = model.forward(x);
			
			float batchAcc = acuracia.forward(prev, y).item();
			int batchSize = end - start;
			hits += batchAcc * batchSize;
		}

		return new Tensor(1).set(hits / n, 0);
	}

	/**
	 * Calcula a precisão em relação aos dados de entrada e saída fornecidos.
	 * @param loader {@code DataLoader} contendo dataset de teste.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor accuracy(DataLoader loader) {
		return accuracy(loader.getX(), loader.getY());
	}

	/**
	 * Calcula a matriz de confusão.
	 * <p>
	 *		A matriz de confusão mostra a contagem de amostras que foram 
	 * 		classificadas de forma ccorreta ou não em cada classe. As linhas 
	 *		representam as classes reais e as colunas as classes previstas.
	 * </p>
	 * @param xs {@code Tensores} com dados de entrada para o modelo.
	 * @param ys {@code Tensores} com dados reais.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor confusionMatrix(Tensor[] xs, Tensor[] ys) {
		Tensor[] prevs = model.forward(xs);
		return mc.forward(prevs, ys);
	}

	/**
	 * Calcula a matriz de confusão a partir de um conjunto de dados.
	 * <p>
	 *		A matriz de confusão mostra a contagem de amostras que foram 
	 * 		classificadas de forma ccorreta ou não em cada classe. As linhas 
	 *		representam as classes reais e as colunas as classes previstas.
	 * </p>
	 * @param loader {@code DataLoader} contendo dataset de teste.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor confusionMatrix(DataLoader loader) {
		return confusionMatrix(loader.getX(), loader.getY());
	}

	/**
	 * Calcula o F1-Score ponderado.
	 * <p>
	 *		O F1-Score é uma métrica que combina a precisão e o recall para 
	 *		avaliar o desempenho de um modelo de classificação. Ele é especialmente 
	 *		útil quando se lida com classes desbalanceadas ou quando se deseja equilibrar 
	 *		a precisão e o recall.
	 * </p>
	 * @param xs {@code Tensores} com dados de entrada.
	 * @param ys {@code Tensores} com dados reais relacionados a entrada.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor f1Score(Tensor[] xs, Tensor[] ys) {
		Tensor[] prevs = model.forward(xs);
		return f1Score.forward(prevs, ys);
	}

	/**
	 * Calcula o F1-Score ponderado.
	 * <p>
	 *		O F1-Score é uma métrica que combina a precisão e o recall para 
	 *		avaliar o desempenho de um modelo de classificação. Ele é especialmente 
	 *		útil quando se lida com classes desbalanceadas ou quando se deseja equilibrar 
	 *		a precisão e o recall.
	 * </p>
	 * @param loader {@code DataLoader} contendo dataset de teste.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor f1Score(DataLoader loader) {
		return f1Score(loader.getX(), loader.getY());
	}
}
