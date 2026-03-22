package jnn.camadas;

import jnn.core.JNNutils;
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
public class BatchNorm2D extends Camada implements Cloneable {

    /**
     * Parâmetro de escala treinável, com formato:
     * <pre>gamma = (canais)</pre>
     */
    public Tensor _gamma;
    
    /**
     * Parâmetro de deslocamento treinável, com fomato:
     * <pre>beta = (canais)</pre>
     */
    public Tensor _beta;

    /**
     * Média móvel acumulada durante o treinamento, utilizada na 
     * fase de inferência, com fomato:
     * <pre>runningMean = (canais)</pre>
     */
    public Tensor _mediaMovel;

    /**
     * Variância móvel acumulada durante o treinamento utilizada na 
     * fase de inferência, com fomato:
     * <pre>runningVar = (canais)</pre>
     */
    public Tensor _varianciaMovel;

    /**
     * Entrada normalizada por canal, com fomato:
     * <pre>entradaNorm = (canais, altura, largura)</pre>
     */
    public Tensor _entradaNorm;

    /**
     * Média calculada de cada lote, com fomato:
     * <pre>media = (canais)</pre>
    */
   public Tensor _media;
   
   /**
    * Variância calculada de cada lote, com fomato:
     * <pre>var = (canais)</pre>
     */
    public Tensor _variancia;

    /**
     * Gradientes da escala, com fomato:
     * <pre>gradGamma = (canais)</pre>
     */
    public Tensor _gradGamma;

    /**
     * Gradientes do deslocamento, com fomato:
     * <pre>gradBeta = (canais)</pre>
     */    
    public Tensor _gradBeta;

	/**
	 * Entrada da camada, com fomato:
	 * <pre>entrada = (canais, altura, largura) </pre>
	 */
    public Tensor _entrada;

	/**
	 * Gradiente em relação a camada, com fomato:
	 * <pre>gradEntrada = (canais, altura, largura) </pre>
	 */
    public Tensor _gradEntrada;

	/**
	 * Saída da camada, com fomato:
	 * <pre>saída = (canais, altura, largura) </pre>
	 */
    public Tensor _saida;
	
    /**
	 * Gradiente em relação a da camada, com fomato:
	 * <pre>gradSaida = (canais, altura, largura) </pre>
	 */
    public Tensor _gradSaida;

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
	 * Auxilar no controle de treinamento em lotes.
	 */
	private int tamLote;

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
    public void construir(int[] shape) {
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

        _gamma = addParam("Gamma", canais);
        _beta = addParam("Beta", canais);
        _gradGamma = addGrad("Grad Gamma", canais);
        _gradBeta = addGrad("Grad Beta", canais);

        _gradEntrada = addBuffer("Grad Entrada", shape);
        _saida = addBuffer("Saida", shape);
        _entradaNorm = addBuffer("Entrada Normalizada", shape);

        _mediaMovel = addBuffer("Media movel", canais);
        _varianciaMovel = addBuffer("Variancia Movel", canais);

        _media = addBuffer("Media", canais);
        _variancia = addBuffer("Variancia", canais);

		_treinavel = true;// camada pode ser treinada.
		_construida = true;// camada pode ser usada.
    }

    @Override
    public void init() {
        verificarConstrucao();
        
        _gamma.preencher(1);
        _beta.preencher(0);

        _mediaMovel.preencher(0);
        _varianciaMovel.preencher(1);
    }

    @Override
    public void ajustarParaLote(int tamLote) {
        int[] shape;
        
        if (tamLote == 0) {
            shape = shapeIn;
            
        } else {
            shape = new int[shapeIn.length + 1];
            shape[0] = tamLote;
            for (int i = 0; i < shapeIn.length; i++) {
                shape[i+1] = shapeIn[i];
            }
        }
        
        _entradaNorm = addBuffer("Entrada Normalizada", shape);
        _saida       = addBuffer("Saida", shape);
        _gradEntrada = addBuffer("Grad Entrada", shape);

        this.tamLote = tamLote;
    }

    @Override
    public Tensor forward(Tensor x) {
        verificarConstrucao();

        boolean entrada3D = x.numDim() == 3;
        if (entrada3D) {
            validarShapes(x.shape(), shapeIn);
            ajustarParaLote(0);

        } else if (x.numDim() == 4) {
            validarShapes(x.shape(), shapeIn);
            int lote = x.tamDim(0);
            if (this.tamLote != lote) ajustarParaLote(lote);  
            
        } else {
            throw new IllegalArgumentException(
                "\nEntrada deve ter 3 (C, H, W) ou 4 dimensões (N, C, H, W), recebido " + x.numDim()
            );
        }
        
        _entrada = x.contiguous();

        lops.forwardBatchNorm2D(
            _entrada,
            _gamma,
            _beta,
            _entradaNorm,
            _mediaMovel,
            _varianciaMovel,
            _media,
            _variancia,
            _saida,
            momentum,
            eps,
            treinando
        );

        return _saida;
    }

    @Override
    public Tensor backward(Tensor g) {
        verificarConstrucao();

        if (!treinando) {
            // se não acumula os gradientes
            throw new IllegalStateException("\nBackward não deve ser chamado em modo avaliação.");
        }

        _gradSaida = g.contiguous();

        lops.backwardBatchNorm2D(
            _gradEntrada,
            _entradaNorm,
            _variancia,
            _gamma,
            _gradGamma,
            _gradBeta,
            _gradSaida,
            eps
        );

        return _gradEntrada;
    }
    
    @Override
    public void gradZero() {
        verificarConstrucao();

        _gradGamma.preencher(0);
        _gradBeta.preencher(0);
    }

    @Override
    public int[] shapeIn() {
        verificarConstrucao();
        return shapeIn.clone();
    }

    @Override
    public int[] shapeOut() {
        return shapeIn();
    }

    @Override
    public int numParams() {
        return _gamma.tam() + _beta.tam();
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
    public Camada clone() {
        try {
            BatchNorm2D clone = (BatchNorm2D) super.clone();
            
            clone._gamma = _gamma.clone();
            clone._beta = _beta.clone();
            clone._mediaMovel = _mediaMovel.clone();
            clone._varianciaMovel = _varianciaMovel.clone();
            clone._entradaNorm = _entradaNorm.clone();
            clone._media = _media.clone();
            clone._variancia = _variancia.clone();
            clone._gradGamma = _gradGamma.clone();
            clone._gradBeta = _gradBeta.clone();
            clone._gradEntrada = _gradEntrada.clone();
            clone._saida = _saida.clone();
            
            clone.shapeIn = shapeIn.clone();
            
            return clone;

        } catch (Exception e) {
            throw e;
        }
    }

}