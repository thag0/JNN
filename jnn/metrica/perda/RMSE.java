package jnn.metrica.perda;

import jnn.core.tensor.Tensor;

/**
 * Função de perda Root Mean Squared Error, calcula a raiz do 
 * erro médio quadrado entre as previsões e os valores reais.
 */
public class RMSE extends Perda {
 
	/**
	 * Inicializa a função de perda Root Mean Squared Error.
	 */
	public RMSE() {}

	@Override
	public Tensor forward(Tensor prev, Tensor real) {
		verificarDimensoes(prev, real);
		
		if (prev.numDim() == 1) {
			int tam = prev.tam();
			double rmse = f(prev, real, tam);
			
			return new Tensor(
				new double[]{ Math.sqrt(rmse / tam) }
			);
		}

        int lotes = prev.tamDim(0);
        int amostras = prev.tamDim(1);

        double somaLote = 0;
        for (int i = 0; i < lotes; i++) {
            Tensor p = prev.subTensor(i);
            Tensor r = real.subTensor(i);

            double soma = f(p, r, amostras) / amostras;
            somaLote += Math.sqrt(soma);
        }

        return new Tensor(
			new double[]{ somaLote / lotes }
		);
	}

	/**
	 * Calculo interno do rmse.
	 * @param prev tensor com dados previstos.
	 * @param real tensor com dados reais.
	 * @param tam quantidade de amostras.
	 * @return soma do rmse.
	 */
	private double f(Tensor prev, Tensor real, int tam) {
		double rmse = 0.0;
		for (int i = 0; i < tam; i++) {
			double d = prev.get(i) - real.get(i);
			rmse += d * d;
		}

		return rmse;
	}
	 
	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		super.verificarDimensoes(prev, real);
		final int tam = prev.tam();
		double rrmse = Math.sqrt(forward(prev, real).item());

		return prev.map(
			real,
			(p, r) -> (p - r) / (rrmse * tam)
		);
	}
}
