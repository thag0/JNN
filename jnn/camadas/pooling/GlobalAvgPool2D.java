package jnn.camadas.pooling;

import jnn.camadas.Camada;
import jnn.camadas.LayerOps;
import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Camada de Agrupamento Médio Global.
 * <p>
 *      A camada GlobalAvgPool é responsável por calcular a média de
 *      cada canal de entrada e poduzir uma única saída por canal, reduzindo
 *      bastante a quantidade de parâmetros em alguns modelos e reduzindo
 *      a chance de overfitting.
 * </p>
 * <p>
 *      Exemplo
 * </p>
 * <pre>
 *x ([[[1.0, 2.0],
 *     [3.0, 4.0]],
 *
 *    [[5.0, 6.0],
 *     [7.0, 8.0]]])
 * 
 * y = gap.forward(x);
 * 
 * y = ([2.5, 6.5])
 * </pre>
 */
public class GlobalAvgPool2D extends Camada implements Cloneable {

    /**
     * Utilitário.
     */
    LayerOps lops = new LayerOps();
    
	/**
	 * Formato de entrada da camada, dado por:
	 * <pre>
	 *    shape = (canais, altura, largura)
	 * </pre>
	 */
    private final int[] shapeIn  = {1, 1, 1};

	/**
	 * Formato de saída da camada, dado por:
	 * <pre>
	 *    shape = (canais)
	 * </pre>
	 */
    private final int[] shapeOut = {1};

	/**
	 * Auxilar no controle de treinamento em lotes.
	 */
    private int tamLote;

	/**
	 * Tensor contendo os valores de entrada para a camada.
	 * <p>
	 *    O formato da entrada é dado por:
	 * </p>
	 * <pre>shape = (canais, altura, largura) </pre>
	 */
    public Tensor _entrada;
    
	/**
	 * Tensor contendo os valores de saída calculados pela camada.
	 * <p>
	 *    O formato de saída é dado por:
	 * </p>
	 * <pre>shape = (canais) </pre>
    */
   public Tensor _saida;
   
    /**
    * Tensor contendo os gradientes em relação a entrada da camada.
     * <p>
     *    Seu formato é dado por:
     * </p>
     * <pre>shape = (canais, altura, largura) </pre>
     */
    public Tensor _gradEntrada;
    
    /**
    * Tensor contendo os gradientes em relação a saída da camada.
     * <p>
     *    Seu formato é dado por:
     * </p>
     * <pre>shape = (canais) </pre>
     */
    public Tensor _gradSaida;

    /**
     * Inicializa uma camada GlobalAvgPool2D, sem especificar o formato de
     * entrada.
     */
    public GlobalAvgPool2D() {}
    
    /**
     * Inicializa uma camada GlobalAvgPool2D.
     * @param shape formato de entrada para a camada.
     */
    public GlobalAvgPool2D(int[] shape) {
        construir(shape);
    }

    @Override
    public void construir(int[] shape) {
        JNNutils.validarNaoNulo(shape, "shape == null.");

		if (shape.length != 3) {
			throw new IllegalArgumentException(
				"\nFormato de entrada para a camada " + nome() + " deve conter três " + 
				"elementos (canais, altura, largura), mas recebido tamanho = " + shape.length
			);
		}

		if (!JNNutils.apenasMaiorZero(shape)) {
			throw new IllegalArgumentException(
				"\nOs valores de dimensões de entrada para a camada " + nome() + 
				" devem ser maiores que zero."
			);
		}

        shapeIn[0] = shape[0];//canais
        shapeIn[1] = shape[1];//altura
        shapeIn[2] = shape[2];//largura

        shapeOut[0] = shapeIn[0];//canais

        _gradEntrada = addBuffer("Grad Entrada", shapeIn);
        _saida       = addBuffer("Saida", shapeOut);

        _treinavel = false;
        _construida = true;
    }

    @Override
    public void inicializar() {}

    @Override
    public void ajustarParaLote(int tamLote) {
        int[] in;
        int[] out;

        if (tamLote == 0) {
            in = shapeIn;
            out = shapeOut;

        } else {
            in = new int[4];
            in[0] = tamLote;
            in[1] = shapeIn[0];
            in[2] = shapeIn[1];
            in[3] = shapeIn[2];

            out = new int[2];
            out[0] = tamLote;
            out[1] = shapeOut[0];
        }

        _gradEntrada = addBuffer("Grad Entrada", in);
        _saida       = addBuffer("Saida", out);

        this.tamLote = tamLote;
    }

    @Override
    public Tensor forward(Tensor x) {
        verificarConstrucao();

        final int numDim = x.numDim();

        if (numDim == 3) {
            ajustarParaLote(0);

        } else if (numDim == 4) {
            int lotes = x.tamDim(0);
            if (lotes != this.tamLote) {
                ajustarParaLote(lotes);
            }

        } else {
            throw new UnsupportedOperationException(
                "Esperado 3D ou 4D, recebido: " + numDim + "D."
            );
        }

        _entrada = x.contiguous();

        lops.forwardGAP(_entrada, _saida, shapeIn);

        return _saida;
    }

    @Override
    public Tensor backward(Tensor g) {
        verificarConstrucao();

        _gradSaida = g.contiguous();
        _gradEntrada.zero();

        lops.backwardGAP(_gradEntrada, _gradSaida, shapeIn, tamLote);

        return _gradEntrada;
    }

    @Override
    public int[] shapeIn() {
        verificarConstrucao();
        return shapeIn.clone();
    }

    @Override
    public int[] shapeOut() {
        verificarConstrucao();
        return shapeOut.clone();
    }

    @Override
    public int numParams() {
        return 0;
    }

    @Override
    public Tensor saida() {
        verificarConstrucao();
        return _saida;
    }

	@Override
	public Tensor gradEntrada() {
		verificarConstrucao();
		return _gradEntrada; 
	}

    @Override
    public long tamBytes() {
		long tamVars = super.tamBytes(); //base camada + tensores
		tamVars += 4 * shapeIn.length; 
		tamVars += 4 * shapeOut.length; 
		tamVars += 4;//tamLote 

		long tamTensores =
		_saida.tamBytes() +
		_gradEntrada.tamBytes();

		return tamVars + tamTensores;
    }

    @Override
    public Camada clone() {
        verificarConstrucao();

        GlobalAvgPool2D clone = (GlobalAvgPool2D) super.clone();

		clone._gradEntrada = this._gradEntrada.clone();
		clone._saida = this._saida.clone();

        return clone;
    }

}
