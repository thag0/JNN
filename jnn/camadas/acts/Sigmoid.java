package jnn.camadas.acts;

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

        sigmoid(
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

        sigmoidd(
            _gradSaida.array(), _gradSaida.offset(),
            _gradEntrada.array(), _gradEntrada.offset(),
            _saida.array(), _saida.offset(),
            _entrada.tam()
        );

        return _gradEntrada;
    }

    private void sigmoid(float[] x, int offX, float[] dest, int offDest, int n) {
        for (int i = 0; i < n; i++) {
            float val = x[offX + i];
            dest[offDest + i] = (float) (1.0f / (1.0 + Math.exp(-val)));
        }
    }

    private void sigmoidd(float[] g, int offG, float[] gradE, int offGE, float[] sig, int offSig, int n) {
        for (int i = 0; i < n; i++) {
            final float grad = g[offG + i];
            final float s = sig[offSig + i];
            gradE[offGE + i] = grad * (s * (1f - s));
        }
    }
    
    @Override
    public Sigmoid clone() {
        return (Sigmoid) super.clone();
    }

}
