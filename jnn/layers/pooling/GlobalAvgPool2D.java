package jnn.layers.pooling;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;
import jnn.layers.Layer;
import jnn.layers.LayerOps;

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
public class GlobalAvgPool2D extends Layer implements Cloneable {

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
	 * Tensor contendo os valores de entrada para a camada.
	 * <p>
	 *    O formato da entrada é dado por:
	 * </p>
	 * <pre>shape = (canais, altura, largura) </pre>
	 */
    public Tensor _input;
    
	/**
	 * Tensor contendo os valores de saída calculados pela camada.
	 * <p>
	 *    O formato de saída é dado por:
	 * </p>
	 * <pre>shape = (canais) </pre>
    */
   public Tensor _output;
   
    /**
    * Tensor contendo os gradientes em relação a entrada da camada.
     * <p>
     *    Seu formato é dado por:
     * </p>
     * <pre>shape = (canais, altura, largura) </pre>
     */
    public Tensor _gradInput;
    
    /**
    * Tensor contendo os gradientes em relação a saída da camada.
     * <p>
     *    Seu formato é dado por:
     * </p>
     * <pre>shape = (canais) </pre>
     */
    public Tensor _gradOutput;

    /**
     * Inicializa uma camada GlobalAvgPool2D, sem especificar o formato de
     * entrada.
     */
    public GlobalAvgPool2D() {}
    
    /**
     * Inicializa uma camada GlobalAvgPool2D.
     * @param input formato de entrada para a camada.
     */
    public GlobalAvgPool2D(int[] input) {
        build(input);
    }

    @Override
    public void build(int[] shape) {
        JNNutils.validarNaoNulo(shape, "shape == null.");

		if (shape.length != 3) {
			throw new IllegalArgumentException(
				"\nFormato de entrada para a camada " + name() + " deve conter três " + 
				"elementos (canais, altura, largura), mas recebido tamanho = " + shape.length
			);
		}

		if (!JNNutils.apenasMaiorZero(shape)) {
			throw new IllegalArgumentException(
				"\nOs valores de dimensões de entrada para a camada " + name() + 
				" devem ser maiores que zero."
			);
		}

        shapeIn[0] = shape[0];//canais
        shapeIn[1] = shape[1];//altura
        shapeIn[2] = shape[2];//largura

        shapeOut[0] = shapeIn[0];//canais

        _gradInput = addBuffer("grad input", shapeIn);
        _output    = addBuffer("output", shapeOut);

        _treinable = false;
        _built     = true;
    }

    @Override
    public void initParams() {}

    @Override
    public void fitToBatch(int batchSize) {
        int[] in;
        int[] out;

        if (batchSize == 0) {
            in = shapeIn;
            out = shapeOut;

        } else {
            in = new int[] {
                batchSize,
                shapeIn[0],
                shapeIn[1],
                shapeIn[2]
            };

            out = new int[] {
                batchSize,
                shapeOut[0]
            };
        }

        _gradInput = addBuffer("grad input", in);
        _output    = addBuffer("output", out);

        this.batchSize = batchSize;
    }

    @Override
    public Tensor forward(Tensor x) {
        isBuilt();

        final int numDim = x.numDim();

        if (numDim == 3) {
            validateShapes(x.shape(), shapeIn);
            if (this.batchSize != 0) fitToBatch(0);
            
        } else if (numDim == 4) {
            validateShapes(x.shape(), shapeIn);
            int lotes = x.lenDim(0);
            if (lotes != this.batchSize) fitToBatch(lotes);

        } else {
            throw new UnsupportedOperationException(
                "Esperado 3D ou 4D, recebido: " + numDim + "D."
            );
        }

        _input = x.contiguous();

        lops.forwardGAP(_input, _output, shapeIn);

        return _output;
    }

    @Override
    public Tensor backward(Tensor g) {
        isBuilt();

        _gradOutput = g.contiguous();

        lops.backwardGAP(_gradInput, _gradOutput, shapeIn, batchSize);

        return _gradInput;
    }

    @Override
    public int[] shapeIn() {
        isBuilt();
        return shapeIn.clone();
    }

    @Override
    public int[] shapeOut() {
        isBuilt();
        return shapeOut.clone();
    }

    @Override
    public int numParams() {
        return 0;
    }

    @Override
    public Tensor output() {
        isBuilt();
        return _output;
    }

	@Override
	public Tensor gradInput() {
		isBuilt();
		return _gradInput; 
	}

    @Override
    public long sizeBytes() {
		long tamVars = super.sizeBytes(); //base camada + tensores
		tamVars += 4 * shapeIn.length; 
		tamVars += 4 * shapeOut.length; 
		tamVars += 4;//tamLote 

		long tamTensores =
		_output.sizeBytes() +
		_gradInput.sizeBytes();

		return tamVars + tamTensores;
    }

    @Override
    public Layer clone() {
        isBuilt();

        GlobalAvgPool2D clone = (GlobalAvgPool2D) super.clone();

		clone._gradInput = this._gradInput.clone();
		clone._output = this._output.clone();

        return clone;
    }

    @Override
    public String info() {
        StringBuilder sb = new StringBuilder();
        String pad = " ".repeat(4);

		sb.append(name()).append("(\n");
        sb.append(pad).append("In: ").append(JNNutils.arrayStr(shapeIn)).append("\n");
        sb.append(pad).append("Out: ").append(JNNutils.arrayStr(shapeOut)).append("\n");
        sb.append(")");

        return sb.toString();
    }

}
