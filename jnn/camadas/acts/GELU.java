package jnn.camadas.acts;

import jnn.core.tensor.Tensor;

/**
 * Camada que opera a função de ativação GELU.
 */
public class GELU extends Ativacao {

    /**
     * Constante pré calculada.
     */
	private final float RAIZ_2_POR_PI = (float) Math.sqrt(2 / Math.PI); 

    /**
     * Constante empírica usada na aproximação da função junto da tanh.
     */
	private final float C = 0.044715f;
    
    /**
     * Inicializa uma camada de ativação GELU.
     * @param shape formato de entrada para a camada.
     */
    public GELU(int... shape) {
        construir(shape);
    }
 
    /**
     * Inicializa uma camada de ativação GELU.
     */
    public GELU () {}

    @Override
    public Tensor forward(Tensor x) {
        verificarConstrucao();

		talvesAjustarLote(x);

		_entrada = x.contiguous();

        gelu(
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

        gelud(
            _gradSaida.array(), _gradSaida.offset(),
            _gradEntrada.array(), _gradEntrada.offset(),
            _entrada.array(), _entrada.offset(),
            _entrada.tam()
        );

        return _gradEntrada;
    }

    private void gelu(float[] x, int offX, float[] dest, int offDest, int n) {
        for (int i = 0; i < n; i++) {
            float v = x[offX + i];
            float v3 = v * v * v;

            float u = RAIZ_2_POR_PI * (v + C * v3);
            float t = tanh(u);

            dest[offDest + i] = 0.5f * v * (1.0f + t);
        }
    }

    private void gelud(float[] g, int offG, float[] gradE, int offGE, float[] x, int offX, int n) {
        for (int i = 0; i < n; i++) {
            float v = x[offX + i];
            float v2 = v * v;
            float v3 = v2 * v;
            float u = RAIZ_2_POR_PI * (v + C * v3);
            float t = tanh(u);
            float dt = 1.0f - t * t;
            float du = RAIZ_2_POR_PI * (1.0f + 3.0f * C * v2);
            float dgelu = 0.5f * (1.0f + t) + 0.5f * v * dt * du;

            gradE[offGE + i] = g[offG + i] * dgelu;
        }
    }

    private float tanh(float x) {
        return (float) (2.0 / (1.0 + Math.exp(-2.0 * x)) - 1.0);
    }

}
