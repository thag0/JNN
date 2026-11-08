package jnn.camadas.acts;

import jnn.camadas.Layer;
import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Camada de ativação base.
 */
public abstract class Activation extends Layer implements Cloneable {
    
    /**
     * Tensor contendo os dados de entrada da camada.
     */
    protected Tensor _input;

    /**
     * Tensor contendo os dados de saída calculados pela camada.
     */
    protected Tensor _output;

    /**
     * Tensor contendo os gradientes em relação a entrada da camada.
     */
    protected Tensor _gradInput;
    
    /**
     * Tensor contendo os gradientes em relação a saída da camada.
     */
    protected Tensor _gradOutput;

    /**
     * Formato de entrada (e saída) da camada.
     */
    protected int[] shapeIn;

    /**
     * Dimensão base da camada usada quando foi construída.
     */
    protected int baseDim;

    /**
     * Construtor privado.
     */
    protected Activation() {}

    @Override
    public void build(int[] shape) {
		JNNutils.validarNaoNulo(shape, "shape == null");

		if (shape.length < 1) {
			throw new IllegalArgumentException(
				"\nO formato deve conter pelo menos um elemento."
			);
		}

		if (!JNNutils.apenasMaiorZero(shape)) {
			throw new IllegalArgumentException(
				"\nValores do formato de entrada devem ser maiores que zero."
			);
		}

        shapeIn = shape.clone();
        baseDim = shape.length;

        _output    = addBuffer("output", shapeIn);
        _gradInput = addBuffer("grad input", shapeIn);

        _built = true;
    }

    @Override
    public void initParams() {}

	@Override
	public void fitToBatch(int batchSize) {
        int[] shape = {};

		if (batchSize == 0) {
            shape = shapeIn;
            
		} else {
            shape = new int[shapeIn.length + 1];
            shape[0] = batchSize;
            for (int i = 0; i < shapeIn.length; i++) {
                shape[i+1] = shapeIn[i];
            }

		}

        _gradInput = addBuffer("Grad Entrada", shape);
        _output = addBuffer("Saida", shape);
		
		this.batchSize = batchSize;
	}

    /**
     * Pode reajustar os parâmetros da camada caso suas dimensões sejam compatíveis.
     * @param x {@code Tensor} base de entrada.
     */
    protected void maybeFitToBatch(Tensor x) {
        int numDims = x.numDim();

        if (numDims == baseDim) {
            validateShapes(x.shape(), shapeIn);
            if (this.batchSize != 0) fitToBatch(0);

        } else if (numDims == baseDim + 1) {
            validateShapes(x.shape(), shapeIn);
            int lote = x.lenDim(0);
            if (lote != this.batchSize) fitToBatch(lote);

        } else {
            throw new UnsupportedOperationException(
                "Esperado tensor com " + baseDim +
                " ou " + (baseDim + 1) +
                " dimensões. Recebido: " + numDims
            );
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        throw new UnsupportedOperationException(
            "\nImplementar forward."
        );
    }

    @Override
    public Tensor backward(Tensor g) {
        throw new UnsupportedOperationException(
            "\nImplementar backward."
        );
    }

    @Override
    public Tensor output() {
        isBuilt();
        return _output;
    }

    @Override
    public int[] shapeIn() {
        return shapeIn;
    }

    @Override
    public int[] shapeOut() {
        return shapeIn();
    }

    @Override
    public int numParams() {
        return 0;
    }

    @Override
    public int sizeOutput() {
        return output().len();
    }

    @Override
    public Tensor gradInput() {
        return _gradInput;
    }

    @Override
    public Activation clone() {
        Activation clone = (Activation) super.clone();

        clone._output = _output.clone();
        clone._gradInput = _gradInput.clone();
        clone.shapeIn = shapeIn.clone();
    
        return clone;
    }

    @Override
    public long sizeBytes() {
        long tamVars = super.sizeBytes();//base
        tamVars += 4 * shapeIn.length;
        tamVars += 4;//tamLote
        tamVars += 4;//dimBase
        
        long tamTensores =
        _gradInput.sizeBytes() +
        _output.sizeBytes();

        return tamVars + tamTensores;
    }

    @Override
    public String info() {
        StringBuilder sb = new StringBuilder();
        String pad = " ".repeat(4);

		sb.append(name()).append("(\n");
        sb.append(pad).append("In: ").append(JNNutils.arrayStr(shapeIn)).append("\n");
        sb.append(")");

        return sb.toString();
    }

}
