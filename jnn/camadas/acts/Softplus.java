package jnn.camadas.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação Softplus.
 */
public class Softplus extends Ativacao {
    
    /**
     * Inicializa uma camada de ativação Softplus.
     */
    public Softplus() {}

    /**
     * Inicializa uma camada de ativação Softplus.
     * @param shape formato de entrada da camada.
     */
    public Softplus(int... shape) {
        construir(shape);
    }

    @Override
    public Tensor forward(Tensor x) {
        verificarConstrucao();

		talvesAjustarLote(x);

		_entrada = x.contiguous();

        softplus(
            _entrada.array(), _entrada.offset(),
            _saida.array(), _saida.offset(),
            _entrada.tam()
        );

        return _saida;
    }

    @Override
    public Tensor backward(Tensor g) {
		verificarConstrucao();

		_gradSaida = g.contiguous();

        softplusd(
            _gradSaida.array(), _gradSaida.offset(),
            _gradEntrada.array(), _gradEntrada.offset(),
            _entrada.array(), _entrada.offset(),
            _entrada.tam()
        );

        return _gradEntrada;
    }

    private void softplus(float[] x, int offX, float[] dest, int offDest, int n) {
        for (int i = 0; i < n; i++) {
            float val = x[offX + i];
            dest[offDest + i] = (float) Math.log(1.0 + Math.exp(val));
        }
    }

    private void softplusd(float[] g, int offG, float[] gradE, int offGE, float[] x, int offX, int n) {
        for (int i = 0; i < n; i++) {
            final float grad = g[offG + i];
            final float entrada = x[offX + i];
            float exp = (float) Math.exp(entrada);
            gradE[offGE + i] = grad * (exp / (1.0f + exp));
        }
    }
    
    @Override
    public Softplus clone() {
        return (Softplus) super.clone();
    }

}
