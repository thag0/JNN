package jnn.camadas.acts;

import jnn.core.JNNnative;
import jnn.core.tensor.Tensor;

/**
 * Camada de ativação ReLU.
 */
public class ReLU extends Ativacao {

    /**
     * Inicializa uma camada de ativação ReLU.
     */
    public ReLU() {}

    /**
     * Inicializa uma camada de ativação ReLU.
     * @param shape formato de entrada para a camada.
     */
    public ReLU(int... shape) {
        construir(shape);
    }

    @Override
    public Tensor forward(Tensor x) {
        verificarConstrucao();

		talvesAjustarLote(x);

		_entrada = x.contiguous();

        if (JNNnative.jni) {
            JNNnative.relu(
                _entrada.array(),
                _saida.array(),
                _entrada.tam()
            );
        } else {
            relu(
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

        if (JNNnative.jni) {
            JNNnative.relud(
                _entrada.array(), 
                _gradSaida.array(), 
                _gradEntrada.array(), 
                _entrada.tam()
            );
        } else {
            relud(
                _entrada.array(),
                _gradSaida.array(), 
                _gradEntrada.array(), 
                _entrada.tam()
            );
        }

        return _gradEntrada;
    }

    private static void relu(float[] x, float[] dst, int n) {
        for (int i = 0; i < n; i++) {
            float val = x[i];
            dst[i] = val > 0 ? val : 0;
        }
    }

    private static void relud(float[] x, float[] g, float[] dst, int n) {
        for (int i = 0; i < n; i++) {
            final float grad = g[i];
            final float entrada = x[i];
            dst[i] = grad * (entrada > 0 ? 1 : 0);
        }
    }
    
    @Override
    public ReLU clone() {
        return (ReLU) super.clone();
    }

}
