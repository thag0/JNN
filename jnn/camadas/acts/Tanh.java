package jnn.camadas.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação Tanh (Tangente Hiperbólica).
 */
public class Tanh extends Ativacao {
    
    /**
     * Inicializa uma camada de ativação Tanh.
     */
    public Tanh() {}
    
    /**
     * Inicializa uma camada de ativação Tanh.
     * @param shape formato de entrada para a camada.
     */
    public Tanh(int... shape) {
        construir(shape);
    }

    @Override
    public Tensor forward(Tensor x) {
        verificarConstrucao();

		talvesAjustarLote(x);

		_entrada = x.contiguous();

        tanh(
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

        tanhd(
            _gradSaida.array(), _gradSaida.offset(),
            _gradEntrada.array(), _gradEntrada.offset(),
            _saida.array(), _saida.offset(),// tanh já calculada
            _entrada.tam()
        );

        return _gradEntrada;
    }

    private void tanh(float[] x, int offX, float[] dest, int offDest, int n) {
        for (int i = 0; i < n; i++) {
            float val = x[offX + i];
            dest[offDest + i] = (float) (2.0 / (1.0 + Math.exp(-2.0 * val)) - 1.0);
        }
    }

    private void tanhd(float[] g, int offG, float[] gradE, int offGE, float[] tanh, int offTanh, int n) {
        for (int i = 0; i < n; i++) {
            final float grad = g[offG + i];
            final float t = tanh[offTanh + i];
            final float d = 1.0f - (t * t);
            gradE[offGE + i] = grad * d;
        }
    }
    
    @Override
    public Tanh clone() {
        return (Tanh) super.clone();
    }

}
