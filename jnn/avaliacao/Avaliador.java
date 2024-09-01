package jnn.avaliacao;

import jnn.avaliacao.metrica.*;
import jnn.avaliacao.perda.*;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;

/**
 * Contém implementações de métricas para analisar modelos.
 */
public class Avaliador {
	private Modelo modelo;

	EntropiaCruzada ecc = new EntropiaCruzada();
	EntropiaCruzadaBinaria ecb = new EntropiaCruzadaBinaria();
	Acuracia acuracia = new Acuracia();
	MatrizConfusao matrizConfusao = new MatrizConfusao();
	F1Score f1Score = new F1Score();
	MatrizConfusao mc = new MatrizConfusao();
	MSE emq = new MSE();
	MAE ema = new MAE();
	MSLE emql = new MSLE();   

	/**
	 * Instancia um novo avaliador destinado a uma Rede Neural.
	 * @param modelo rede neural para o avaliador.
	 * @throws IllegalArgumentException se a rede fornecida for nula.
	 */
	public Avaliador(Modelo modelo) {
		if (modelo == null) {
			throw new IllegalArgumentException("\nO modelo fornecido não pode ser nulo.");
		}

		this.modelo = modelo;
	}

	/**
	 * Calcula o erro médio quadrado em relação aos dados previstos e reais.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor erroMedioQuadrado(Tensor prev, Tensor real) {
		return ema.forward(prev, real);
	}

	/**
	 * Calcula o erro médio quadrado médio em relação aos dados de entrada e 
	 * saída fornecidos.
	 * @param entrada {@code Tensores} com dados de entrada para o modelo previstos.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor erroMedioQuadrado(Tensor[] entrada, Tensor[] real) {
		Tensor[] prev = modelo.forward(entrada);

		int tam = prev.length;
		double res = 0;
		for (int i = 0; i < tam; i++) {
			res += ema.forward(prev[i], real[i]).item();
		}

		return new Tensor(new double[]{ (res/tam) }, 1);
	}

	/**
	 * Calcula o erro médio quadrado logarítimico em relação aos dados 
	 * previstos e reais.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor erroMedioQuadradoLogaritmico(Tensor prev, Tensor real) {
		return emql.forward(prev, real);
	}

	/**
	 * Calcula o erro médio quadrado logarítimico médio em relação aos dados 
	 * de entrada e saída fornecidos.
	 * @param entrada {@code Tensores} com dados de entrada para o modelo.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor erroMedioQuadradoLogaritmico(Tensor[] entrada, Tensor[] real) {
		Tensor[] prev = modelo.forward(entrada);

		int tam = entrada.length;
		double res = 0;
		for (int i = 0; i < tam; i++) {
			res += emql.forward(prev[i], real[i]).item();
		}

		return new Tensor(new double[]{ (res/tam) }, 1); 
	}

	/**
	 * Calcula o erro médio absoluto em relação aos dados previstos e reais.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor erroMedioAbsoluto(Tensor prev, Tensor real) {
		return ema.forward(prev, real);
	}

	/**
	 * Calcula o erro médio absoluto entre as saídas previstas pela rede neural e os valores reais.
	 * @param entrada {@code Tensores} com dados de entrada para o modelo.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor erroMedioAbsoluto(Tensor[] entrada, Tensor[] real) {
		Tensor[] prev = modelo.forward(entrada);

		int tam = entrada.length;
		double res = 0;
		for (int i = 0; i < tam; i++) {
			res += ema.forward(prev[i], real[i]).item();
		}

		return new Tensor(new double[]{ (res/tam) }, 1);
	}

	/**
	 * Calcula a precisão em relação aos dados de entrada e saída fornecidos.
	 * @param entrada {@code Tensores} com dados de entrada para o modelo.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor acuracia(Tensor[] entrada, Tensor[] real) {
		Tensor[] prevs = modelo.forward(entrada); 
		return acuracia.calcular(prevs, real);
	}

	/**
	 * Calcula a entropia cruzada em relação aos dados previstos e reais.
	 * e as saídas reais fornecidas.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor entropiaCruzada(Tensor previsto, Tensor real) {  
		return ecc.forward(previsto, real);
	}

	/**
	 * Calcula a entropia cruzada entre as saídas previstas pela rede neural
	 * e as saídas reais fornecidas.
	 * @param entrada {@code Tensores} com dados de entrada para o modelo.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor entropiaCruzada(Tensor[] entrada, Tensor[] real) {  
		Tensor[] previsoes = modelo.forward(entrada);

		int tam = entrada.length;
		double res = 0;
		for (int i = 0; i < tam; i++) {
			res += ecc.forward(previsoes[i], real[i]).item();
		}

		return new Tensor(new double[]{ (res/tam) }, 1);
	}

	/**
	 * Calcula a entropia cruzada binária entre as saídas previstas pela rede neural
	 * e as saídas reais fornecidas.
	 * @param prev {@code Tensor} com dados previstos.
	 * @param real {@code Tensor} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor entropiaCruzadaBinaria(Tensor previsto, Tensor real) {
		return ecb.forward(previsto, real);
	}

	/**
	 * Calcula a entropia cruzada binária entre as saídas previstas pela rede neural
	 * e as saídas reais fornecidas.
	 * @param entrada {@code Tensores} com dados de entrada para o modelo.
	 * @param real {@code Tensores} com dados reais.
	 * @return {@code Tensor} contendo o resultado.
	 */
	public Tensor entropiaCruzadaBinaria(Tensor[] entrada, Tensor[] real) {
		Tensor[] prev = modelo.forward(entrada);

		int tam = entrada.length;
		double res = 0;
		for (int i = 0; i < tam; i++) {
			res += ecb.forward(prev[i], real[i]).item();
		}

		return new Tensor(new double[]{ (res/tam) }, 1);
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
		return mc.calcular(prevs, saidas);
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
		return f1Score.calcular(prevs, real);
	}
}
