package jnn.metrica;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;
import jnn.dataloader.DataLoader;
import jnn.metrica.metrica.*;
import jnn.metrica.perda.*;
import jnn.modelos.Modelo;

/**
 * Contém implementações de métricas para analisar modelos.
 */
public class Avaliador {

	/**
	 * Modelo base de avaliação.
	 */
	private Modelo modelo;

	MSE mse = new MSE();
	MAE mae = new MAE();
	MSLE msle = new MSLE();
	RMSE rmse = new RMSE();
	EntropiaCruzada ce = new EntropiaCruzada();
	EntropiaCruzadaBinaria bce = new EntropiaCruzadaBinaria();
	
	MatrizConfusao mc = new MatrizConfusao();
	Acuracia acuracia = new Acuracia();
	F1Score f1Score = new F1Score();

	/**
	 * Instancia um novo avaliador destinado a um Modelo.
	 * @param modelo modelo base.
	 */
	public Avaliador(Modelo modelo) {
		if (modelo == null) {
			throw new IllegalArgumentException(
				"\nModelo == null."
			);
		}

		this.modelo = modelo;
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
	public Tensor acuracia(Tensor[] xs, Tensor[] ys) {
		final int n = xs.length;
		int batch = 32;
		
		float acertos = 0;
		for (int i = 0; i < n; i += batch) {
			int inicio = i;
			int fim = Math.min(inicio + batch, n);
			Tensor[] x = JNNutils.subArray(xs, inicio, fim);
			Tensor[] y = JNNutils.subArray(ys, inicio, fim);
			Tensor[] prev = modelo.forward(x);
			
			float accLote = acuracia.forward(prev, y).item();
			int tamLote = fim - inicio;
			acertos += accLote * tamLote;
		}
		
		return new Tensor(
			new float[] { acertos / n }
		);
	}

	/**
	 * Calcula a precisão em relação aos dados de entrada e saída fornecidos.
	 * @param loader {@code DataLoader} contendo dataset de teste.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor acuracia(DataLoader loader) {
		return acuracia(loader.getX(), loader.getY());
	}

	/**
	 * Calcula a matriz de confusão.
	 * <p>
	 *		A matriz de confusão mostra a contagem de amostras que foram 
	 * 		classificadas de forma ccorreta ou não em cada classe. As linhas 
	 *		representam as classes reais e as colunas as classes previstas.
	 * </p>
	 * @param prev {@code Tensores} com dados de entrada para o modelo.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor matrizConfusao(Tensor[] xs, Tensor[] ys) {
		Tensor[] prevs = modelo.forward(xs);
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
	public Tensor matrizConfusao(DataLoader loader) {
		return matrizConfusao(loader.getX(), loader.getY());
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
		Tensor[] prevs = modelo.forward(xs);
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
