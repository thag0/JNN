package jnn.metrica.perda;

import jnn.core.tensor.Tensor;

/**
 * Função de perda  Mean Squared Logarithmic Error, calcula o 
 * erro médio quadrado logarítmico entre as previsões e os 
 * valores reais.
 */
public class MSLE extends Perda {
	double eps = 1e-8;//evitar log 0

	/**
	 * Inicializa a função de perda Mean Squared Logarithmic Error.
	 */
	public MSLE() {}

	@Override
	public Tensor forward(Tensor prev, Tensor real) {
		verificarDimensoes(prev, real);
		
		if (prev.numDim() == 1) {
			int tam = prev.tam();
			double emql = f(prev, real, tam);
			
			return new Tensor(
				new double[]{ (emql/tam) }
			);
		}

        int lotes = prev.tamDim(0);
        int amostras = prev.tamDim(1);

        double somaLote = 0;
        for (int i = 0; i < lotes; i++) {
            Tensor p = prev.subTensor(i);
            Tensor r = real.subTensor(i);

            somaLote += f(p, r, amostras) / amostras;
        }

        return new Tensor(new double[]{ somaLote / lotes });
	}

	/**
	 * Calculo interno do msle.
	 * @param prev tensor com dados previstos.
	 * @param real tensor com dados reais.
	 * @param tam quantidade de amostras.
	 * @return soma do msle.
	 */
	private double f(Tensor prev, Tensor real, int tam) {
		double emql = 0;
		for (int i = 0; i < tam; i++) {
			double d = Math.log(1.0 +  prev.get(i)) - Math.log(1.0 + real.get(i));
			emql += d * d;
		}
		
		return emql;
	}
	
	@Override
	public Tensor backward(Tensor prev, Tensor real) {
		verificarDimensoes(prev, real);
        
		final int tam = prev.numDim() == 1 ? prev.tamDim(0) : prev.tamDim(1);

        return prev.map(real, (p, r) -> {
            double lp = Math.log(1.0 + p + eps);
            double lr = Math.log(1.0 + r + eps);
            return (2.0 / tam) * (lp - lr) * (1.0 / (1.0 + p + eps));
        });
	}
}
