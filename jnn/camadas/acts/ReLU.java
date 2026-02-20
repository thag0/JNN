package jnn.camadas.acts;

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

        relu(
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

        relud(
            _gradSaida.array(), _gradSaida.offset(),
            _gradEntrada.array(), _gradEntrada.offset(),
            _entrada.array(), _entrada.offset(),
            _entrada.tam()
        );

        return _gradEntrada;
    }

    private void relu(float[] x, int offX, float[] dest, int offDest, int n) {
        for (int i = 0; i < n; i++) {
            float val = x[offX + i];
            dest[offDest + i] = val > 0 ? val : 0;
        }
    }

    private void relud(float[] g, int offG, float[] gradE, int offGE, float[] x, int offX, int n) {
        for (int i = 0; i < n; i++) {
            final float grad = g[offG + i];
            final float entrada = x[offX + i];
            gradE[offGE + i] = grad * (entrada > 0 ? 1 : 0);
        }
    }
    
    @Override
    public ReLU clone() {
        return (ReLU) super.clone();
    }

}
