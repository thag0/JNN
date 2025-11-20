package jnn.metrica;

import jnn.core.Utils;
import jnn.core.tensor.Tensor;
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

	/**
	 * Utilitário.
	 */
	private Utils utils;

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
			utils.concatenar(prev),
			utils.concatenar(real)
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
			utils.concatenar(prev),
			utils.concatenar(real)
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
			utils.concatenar(prev),
			utils.concatenar(real)
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
			utils.concatenar(prev),
			utils.concatenar(real)
		);
	}

	/**
	 * Calcula a precisão em relação aos dados de entrada e saída fornecidos.
	 * @param xs {@code Tensores} com dados de entrada para o modelo.
	 * @param ys {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor acuracia(Tensor[] xs, Tensor[] ys) {
		final int amostras = xs.length;
		double acc = 0;

		for (int i = 0; i < amostras; i++) {
			Tensor prev = modelo.forward(xs[i]);
			Tensor val = acuracia.forward(new Tensor[] {prev}, new Tensor[]{ys[i]});
			acc += val.item();
		}
		
		return new Tensor(
			new double[] { acc / amostras }
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
			utils.concatenar(prev),
			utils.concatenar(real)
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
			utils.concatenar(prev),
			utils.concatenar(real)
		);
	}

	/**
	 * Calcula a matriz de confusão para avaliar o desempenho da rede em classificação.
	 * <p>
	 *    A matriz de confusão mostra a contagem de amostras que foram classificadas de forma 
	 *    correta ou não em cada classe. As linhas representam as classes reais e as colunas as 
	 *    classes previstas pela rede.
	 * </p>
	 * @param entrada {@code Tensores} com dados de entrada para o modelo.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor matrizConfusao(Tensor[] entrada, Tensor[] saidas) {
		Tensor[] prevs = modelo.forward(entrada);
		return mc.forward(prevs, saidas);
	}

	/**
	 * Calcula o F1-Score ponderado para o modelo de rede neural em relação às entradas e 
	 * saídas fornecidas.
	 * <p>
	 *    O F1-Score é uma métrica que combina a precisão e o recall para avaliar o desempenho 
	 *    de um modelo de classificação. Ele é especialmente útil quando se lida com classes 
	 *    desbalanceadas ou quando se deseja equilibrar a precisão e o recall.
	 * </p>
	 * @param entrada {@code Tensores} com dados de entrada para o modelo.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor f1Score(Tensor[] entrada, Tensor[] real) {
		Tensor[] prevs = modelo.forward(entrada);
		return f1Score.forward(prevs, real);
	}
}
