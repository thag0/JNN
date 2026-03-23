package jnn.camadas.acts;

import jnn.core.JNNnative;
import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação Sigmoid.
 */
public class Sigmoid extends Ativacao {

    /**
     * Inicializa uma camada de ativação Sigmoid.
     */
    public Sigmoid() {}
    
    /**
     * Inicializa uma camada de ativação Sigmoid.
     * @param shape formato de entrada para a camada.
     */
    public Sigmoid(int... shape) {
        construir(shape);
    }

    @Override
    public Tensor forward(Tensor x) {
        verificarConstrucao();

		talvesAjustarLote(x);

		_entrada = x.contiguous();

        if (JNNnative.isOn()) {
            JNNnative.sigmoid(
                _entrada.array(),
                _saida.array(),
                _entrada.tam()
            );
        } else {
            sigmoid(
                _entrada.array(),
                _saida.array(),
                _entrada.tam()
            );
        }

        return _saida;
    }

    @Override
    public Tensor backward(Tensor g) {
		verificarConstrucao();

		_gradSaida = g.contiguous();

        if (JNNnative.isOn()) {
            JNNnative.sigmoidd(
                _saida.array(),
                _gradSaida.array(),
                _gradEntrada.array(),
                _entrada.tam()
            );
        } else {
            sigmoidd(
                _saida.array(),
                _gradSaida.array(),
                _gradEntrada.array(),
                _entrada.tam()
            );
        }

        return _gradEntrada;
    }

    private void sigmoid(float[] x, float[] dest, int n) {
        for (int i = 0; i < n; i++) {
            float val = x[i];
            dest[i] = (float) (1.0f / (1.0 + Math.exp(-val)));
        }
    }

    private void sigmoidd(float[] sig, float[] g, float[] gradE, int n) {
        for (int i = 0; i < n; i++) {
            final float grad = g[i];
            final float s = sig[i];
            gradE[i] = grad * (s * (1f - s));
        }
    }
    
    @Override
    public Sigmoid clone() {
        return (Sigmoid) super.clone();
    }

}
