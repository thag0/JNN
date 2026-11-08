package jnn.layers;

import jnn.core.JNNutils;
import jnn.core.Parametro;
import jnn.core.tensor.Tensor;

/**
 * <h2>
 *     Camada Batch Normalization 2D
 * </h2>
 * <p>
 *     Aplica a técnica de normalização em lote (Batch Normalization) sobre
 *     tensores tridimensionais no formato espacial (canais, altura, largura).
 *     A normalização é realizada por canal, considerando todas as posições
 *     espaciais e amostras do lote.
 * </p>
 * <p>
 *     Essa técnica reduz o deslocamento interno de covariância (internal
 *     covariate shift), estabilizando e acelerando o treinamento de redes
 *     profundas.
 * </p>
 * @see <a href="https://arxiv.org/abs/1502.03167">Paper</a>
 */
public class BatchNorm2D extends Layer implements Cloneable {

    /**
     * Parâmetro de escala treinável, com formato:
     * <pre>gamma = (canais)</pre>
     */
    public Parametro _gamma;
    
    /**
     * Parâmetro de deslocamento treinável, com fomato:
     * <pre>beta = (canais)</pre>
     */
    public Parametro _beta;

    /**
     * Média móvel acumulada durante o treinamento, utilizada na 
     * fase de inferência, com fomato:
     * <pre>runningMean = (canais)</pre>
     */
    public Tensor _runningMean;

    /**
     * Variância móvel acumulada durante o treinamento utilizada na 
     * fase de inferência, com fomato:
     * <pre>runningVar = (canais)</pre>
     */
    public Tensor _runningVar;

    /**
     * Entrada normalizada por canal, com fomato:
     * <pre>entradaNorm = (canais, altura, largura)</pre>
     */
    public Tensor _normInput;

    /**
     * Média calculada de cada lote, com fomato:
     * <pre>media = (canais)</pre>
    */
    public Tensor _mean;
   
   /**
    * Variância calculada de cada lote, com fomato:
     * <pre>var = (canais)</pre>
     */
    public Tensor _variance;

	/**
	 * Entrada da camada, com fomato:
	 * <pre>entrada = (canais, altura, largura) </pre>
	 */
    public Tensor _input;

	/**
	 * Gradiente em relação a camada, com fomato:
	 * <pre>gradEntrada = (canais, altura, largura) </pre>
	 */
    public Tensor _gradInput;

	/**
	 * Saída da camada, com fomato:
	 * <pre>saída = (canais, altura, largura) </pre>
	 */
    public Tensor _output;
	
    /**
	 * Gradiente em relação a da camada, com fomato:
	 * <pre>gradSaida = (canais, altura, largura) </pre>
	 */
    public Tensor _gradOutput;

    /**
     * Pequeno valor para melhorar estabilidade e evitar divisão por zero.
     */
    private float eps;

    /**
     * Fator de suavização usado na atualização das estatísticas móveis.
     */
    private float momentum;
    
    /**
     * Formato de entrada base da camada, dado por:
	 * <pre>shapeIn = (canais, altura, largura) </pre>
     */
    private int[] shapeIn = {1, 1, 1};

    /**
     * Utilitário.
     */
    LayerOps lops = new LayerOps();

    /**
     * Inicializa uma camada BatchNorm2D.
     * @param momentum fator de atualização das estatísticas móveis.
     * @param eps pequeno valor para estabilidade numérica e evitar divisão por zero.
     */
    public BatchNorm2D(Number momentum, Number eps) {
        float m = momentum.floatValue();
        float e = eps.floatValue();

        if (m < 0) {
            throw new IllegalArgumentException(
                "\nValor de momentum deve ser maior que zero, recebido " + m
            );
        }

        if (e < 0) {
            throw new IllegalArgumentException(
                "\nValor de epsilon deve ser maior que zero, recebido " + e
            );
        }

        this.momentum = m;
        this.eps = e;
    }

    /**
     * Inicializa uma camada BatchNorm2D.
     */
    public BatchNorm2D() {
        this(0.1, 1e-5);
    }

    @Override
    public void build(int[] shape) {
        JNNutils.validarNaoNulo(shape, "shape == null");

        if (shape.length != 3) {
            throw new IllegalArgumentException("\nShape deve ter 3 dimensões (C, H, W)");
        }

        if (!JNNutils.apenasMaiorZero(shape)) {
            throw new IllegalArgumentException("\nShape deve conter valores maiores que zero");
        }

        shapeIn[0] = shape[0];//canais
        shapeIn[1] = shape[1];//altura
        shapeIn[2] = shape[2];//largura

        int canais = shapeIn[0];
        
        _gamma = addParam("gamma", canais);
        _beta = addParam("beta", canais);

        _gradInput  = addBuffer("grad input", shape);
        _output     = addBuffer("output", shape);
        _normInput  = addBuffer("norm input", shape);

        _runningMean = addBuffer("running mean", canais);
        _runningVar  = addBuffer("running var",  canais);

        _mean = addBuffer("mean", canais);
        _variance = addBuffer("variance", canais);// melhor não colocar "var"

		_treinable = true;// camada pode ser treinada.
		_built      = true;// camada pode ser usada.
    }

    @Override
    public void initParams() {
        isBuilt();
        
        _gamma.weight.fill(1);
        _beta.weight.fill(0);

        _runningMean.fill(0);
        _runningVar.fill(1);
    }

    @Override
    public void fitToBatch(int batchSize) {
        int[] shape;
        
        if (batchSize == 0) {
            shape = shapeIn;
            
        } else {
            shape = new int[shapeIn.length + 1];
            shape[0] = batchSize;
            for (int i = 0; i < shapeIn.length; i++) {
                shape[i+1] = shapeIn[i];
            }
        }
        
        _normInput = addBuffer("norm input", shape);
        _output    = addBuffer("output", shape);
        _gradInput = addBuffer("grad input", shape);

        this.batchSize = batchSize;
    }

    @Override
    public Tensor forward(Tensor x) {
        isBuilt();

        boolean isInput3d = x.numDim() == 3;
        if (isInput3d) {
            validateShapes(x.shape(), shapeIn);
            fitToBatch(0);

        } else if (x.numDim() == 4) {
            validateShapes(x.shape(), shapeIn);
            int lote = x.lenDim(0);
            if (this.batchSize != lote) fitToBatch(lote);  
            
        } else {
            throw new IllegalArgumentException(
                "\nEntrada deve ter 3 (C, H, W) ou 4 dimensões (N, C, H, W), recebido " + x.numDim()
            );
        }
        
        _input = x.contiguous();

        lops.forwardBatchNorm2D(
            _input,
            _gamma,
            _beta,
            _normInput,
            _runningMean,
            _runningVar,
            _mean,
            _variance,
            _output,
            momentum,
            eps,
            _training
        );

        return _output;
    }

    @Override
    public Tensor backward(Tensor g) {
        isBuilt();

        _gradOutput = g.contiguous();

        lops.backwardBatchNorm2D(
            _gradInput,
            _normInput,
            _variance,
            _gamma,
            _beta,
            _gradOutput,
            eps
        );

        return _gradInput;
    }
    
    @Override
    public void zeroGrad() {
        isBuilt();

        _gamma.grad.fill(0);
        _beta.grad.fill(0);
    }

    @Override
    public int[] shapeIn() {
        isBuilt();
        return shapeIn.clone();
    }

    @Override
    public int[] shapeOut() {
        return shapeIn();
    }

    @Override
    public int numParams() {
        return _gamma.weight.len() + _beta.weight.len();
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

    /**
     * Retorna o valor de epsilon usado.
     * @return epsilon.
     */
    public float eps() {
        return eps;
    }

    /**
     * Retorna o valor de momentum usado.
     * @return momentum.
     */
    public float momentum() {
        return momentum;
    }

    @Override
    public Layer clone() {
        try {
            BatchNorm2D clone = (BatchNorm2D) super.clone();
            
            clone._gamma = _gamma.clone();
            clone._beta = _beta.clone();

            clone._runningMean = _runningMean.clone();
            clone._runningVar = _runningVar.clone();
            clone._normInput = _normInput.clone();
            clone._mean = _mean.clone();
            clone._variance = _variance.clone();
            clone._gradInput = _gradInput.clone();
            clone._output = _output.clone();
            
            clone.shapeIn = shapeIn.clone();
            
            return clone;

        } catch (Exception e) {
            throw e;
        }
    }

    @Override
    public boolean hasBias() {
        return true; //beta funciona como bias
    }

	@Override
	public String info() {
		isBuilt();

		StringBuilder sb = new StringBuilder();
		String pad = " ".repeat(4);
		
		sb.append(name()).append("(\n");

		sb.append(pad).append("In: ").append(JNNutils.arrayStr(shapeIn)).append("\n");
        sb.append(pad).append("Momentum: ").append(momentum).append("\n");

		sb.append(pad).append(_gamma).append("\n");
        sb.append(pad).append(_beta).append("\n");

		sb.append(")");

		return sb.toString();
	}

}