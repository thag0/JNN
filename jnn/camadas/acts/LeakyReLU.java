package jnn.camadas.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada de ativação Leaky ReLU.
 */
public class LeakyReLU extends Ativacao {

    /**
     * Parâmetro personalizável.
     */
    private float alpha;

    /**
     * Inicializa uma camada de ativação Leaky ReLU.
     * @param alfa constante da ativação.
     * @param shape formato de entrada para a camada.
     */
    public LeakyReLU(Number alfa, int... shape) {
        this(alfa);
        construir(shape);
    }

    /**
     * Inicializa uma camada de ativação Leaky ReLU.
     * @param shape formato de entrada para a camada.
     */
    public LeakyReLU(int... shape) {
        construir(shape);
    }

    /**
     * Inicializa uma camada de ativação Leaky ReLU.
     * @param alfa constante da ativação.
     */
    public LeakyReLU(Number alfa) {
        this.alpha = alfa.floatValue();
    }

    /**
     * Inicializa uma camada de ativação Leaky ReLU.
     */
    public LeakyReLU() {
        this(0.01);
    }

    @Override
    public Tensor forward(Tensor x) {
        verificarConstrucao();

        talvesAjustarLote(x);

		_entrada = x.contiguous();

        leakyrelu(
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

        leakyrelud(
            _gradSaida.array(), _gradSaida.offset(),
            _gradEntrada.array(), _gradEntrada.offset(),
            _entrada.array(), _entrada.offset(),
            _entrada.tam()
        );

        return _gradEntrada;
    }

    private void leakyrelu(float[] x, int offX, float[] dest, int offDest, int n) {
        for (int i = 0; i < n; i++) {
            float val = x[offX + i];
            dest[offDest + i] = val > 0 ? val : val * alpha;
        }
    }

    private void leakyrelud(float[] g, int offG, float[] gradE, int offGE, float[] x, int offX, int n) {
        for (int i = 0; i < n; i++) {
            final float grad = g[offG + i];
            final float entrada = x[offX + i];
            gradE[offGE + i] = grad * (entrada > 0f ? 1f : alpha);
        }
    }
    
    @Override
    public LeakyReLU clone() {
        return (LeakyReLU) super.clone();
    }
    
    /**
     * Retorna o parâmetro da camada.
     * @return constante alpha.
     */
    public float getAlpha() {
        return alpha;
    }

}
