package jnn.camadas.pooling;

import jnn.camadas.Camada;
import jnn.camadas.LayerOps;
import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Camada base para facilitar novas implementações de operações
 * personalizadas de pooling.
 */
public abstract class Pool2DBase extends Camada {

	/**
	 * Utilitário.
	 */
	LayerOps lops = new LayerOps();

    /**
     * Formato de entrada da camada em {@code (canais, altura, largura)}
     */
    protected int[] shapeIn = {1, 1, 1};

    /**
     * Formato de saída da camada em {@code (canais, altura, largura)}
     */
    protected int[] shapeOut = {1, 1, 1};

	/**
	 * Auxilar no controle de treinamento em lotes.
	 */
	protected int _tamLote;

	/**
	 * Tensor contendo os dados de entrada da camada.
	 * <p>
	 *    O formato da entrada é dado por:
	 * </p>
	 * <pre>
	 *    entrada = (canais, altura, largura)
	 * </pre>
	 */
	public Tensor _entrada;

	/**
	 * Tensor contendo os dados de saída da camada.
	 * <p>
	 *    O formato de entrada varia dependendo da configuração da
	 *    camada (filtro, strides) mas é dado como:
	 * </p>
	 * <pre>
	 *largura = (larguraEntrada = larguraFiltro) / larguraStride + 1;
	 *altura = (alturaEntrada = alturaFiltro) / alturaStride + 1;
	 * </pre>
	 * <p>
	 *    Com isso o formato de saída é dado por:
	 * </p>
	 * <pre>
	 *    saida = (canais, altura, largura)
	 * </pre>
	 * Essa relação é válida pra cada canal de entrada.
	 */
	public Tensor _saida;

	/**
	 * Tensor contendo os gradientes que serão
	 * retropropagados para as camadas anteriores.
	 * <p>
	 *    O formato do gradiente de entrada é dado por:
	 * </p>
	 * <pre>
	 *    entrada = (canaisEntrada, alturaEntrada, larguraEntrad)
	 * </pre>
	 */
	public Tensor _gradEntrada;

	/**
	 * Formato do filtro de pooling (altura, largura).
	 */
	protected int[] _filtro;

	/**
	 * Valores de stride (altura, largura).
	 */
	protected int[] _stride;

    /**
     * Identificador de operação (max, avg)
     */
    protected String modo;

    /**
     * Construtor interno.
     * @param filtro formato do filtro {@code altura, largura}.
     * @param stride formato dos strides {@code altura, largura}.
     */
    protected Pool2DBase(int[] filtro, int[] stride) {
		JNNutils.validarNaoNulo(filtro, "filtro == null.");
        JNNutils.validarNaoNulo(stride, "stride == null.");

		if (filtro.length != 2) {
			throw new IllegalArgumentException(
				"\nO formato do filtro deve conter dois elementos (altura, largura)."
			);
		}

		if (!JNNutils.apenasMaiorZero(filtro)) {
			throw new IllegalArgumentException(
				"\nOs valores de dimensões do filtro devem ser maiores que zero."
			);
		}

		if (stride.length != 2) {
			throw new IllegalArgumentException(
				"\nO formato para os strides deve conter dois elementos (altura, largura)."
			);
		}

		if (!JNNutils.apenasMaiorZero(stride)) {
			throw new IllegalArgumentException(
				"\nOs valores para os strides devem ser maiores que zero."
			);
		}

		this._filtro = filtro.clone();
		this._stride = stride.clone();
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

		shapeIn[0] = shape[0];// canais
		shapeIn[1] = shape[1];// altura
		shapeIn[2] = shape[2];// largura

		shapeOut[0] = shapeIn[0];
		shapeOut[1] = (int) Math.floor((float)(shapeIn[1] - _filtro[0]) / _stride[0]) + 1;
		shapeOut[2] = (int) Math.floor((float)(shapeIn[2] - _filtro[1]) / _stride[1]) + 1;
		
		if (shapeOut[1] < 1 || shapeOut[2] < 1) {
			throw new IllegalArgumentException(
				"\nCamada não pode ser construida:" +
				"\nFormato de entrada " + JNNutils.arrayStr(shape) +
				" e formato dos filtros " + JNNutils.arrayStr(_filtro) +
				" resultam num formato de saída inválido " + JNNutils.arrayStr(shapeOut)
			);
		}
		
		_gradEntrada = addBuffer("Grad Entrada", shapeIn);
		_saida 		 = addBuffer("Saida", shapeOut);

		_construida = true;// camada pode ser usada
    }

    @Override
    public void inicializar() {}

	@Override
	public void ajustarParaLote(int tamLote) {
		if (tamLote == 0) {
			_gradEntrada = addBuffer("Grad Entrada", shapeIn);
			_saida = addBuffer("Saida", shapeOut);
			
		} else {
			final int canais = shapeIn[0];
			final int altIn = shapeIn[1];
			final int largIn = shapeIn[2];
			final int altOut = shapeOut[1];
			final int largOut = shapeOut[2];
			
			_gradEntrada = addBuffer("Grad Entrada", tamLote, canais, altIn, largIn);
			_saida = addBuffer("Saida", tamLote, canais, altOut, largOut);
		}


		this._tamLote = tamLote;
	}

    @Override
    public Tensor forward(Tensor x) {
		verificarConstrucao();

		final int numDim = x.numDim();

		if (numDim == 3) {
			validarShapes(x.shape(), shapeIn);
			if (_tamLote != 0) ajustarParaLote(0);
		
		} else if (numDim == 4) {
			validarShapes(x.shape(), shapeIn);
			int lotes = x.tamDim(0);
			if (lotes != this._tamLote) ajustarParaLote(lotes);
		
		} else {
			throw new UnsupportedOperationException(
				"Esperado tensor com " + shapeIn.length +
				" ou " + (shapeIn.length + 1) +
				" dimensões. Recebido: " + x.numDim()
			);
		}

		_entrada = x.contiguous();

        if (modo.equals("avg")) {
            lops.forwardAvgPool2D(_entrada, _saida, _filtro, _stride);
        
		} else if (modo.equals("max")) {
            lops.forwardMaxPool2D(_entrada, _saida, _filtro, _stride);
        
        } else {
            throw new UnsupportedOperationException(
                "\nModo \"" + modo + "\" sem suporte."
            );
        }

		return _saida;
    }

    @Override
    public Tensor backward(Tensor g) {
		verificarConstrucao();

		if (g.numDim() != _gradEntrada.numDim()) {
			throw new IllegalStateException(
				"\nEsperado gradiente " + _gradEntrada.numDim() + "D, " +
				" mas recebido " + g.numDim() + "D."
			);
		}

		_gradEntrada.zero();// limpar acumulações anteriores

        if (modo.equals("avg")) {
            lops.backwardAvgPool(_entrada, g, _gradEntrada, _filtro, _stride);
        
        } else if (modo.equals("max")) {
            lops.backwardMaxPool2D(_entrada, g, _gradEntrada, _filtro, _stride);
        
        } else {
            throw new UnsupportedOperationException(
                "\nModo \"" + modo + "\" sem suporte."
            ); 
        }

		return _gradEntrada;
    }

    @Override
    public Tensor saida() {
        verificarConstrucao();
		return _saida;
    }

    @Override
    public int[] shapeIn() {
		verificarConstrucao();
		return shapeIn;
    }

    @Override
    public int[] shapeOut() {
        verificarConstrucao();
		return shapeOut;
    }

    @Override
    public int numParams() {
        return 0;
    }

	/**
	 * Retorna o formato do filtro (altura, largura) usado pela camada.
	 * @return formato do filtro da camada.
	 */
	public int[] formatoFiltro() {
		verificarConstrucao();
		return _filtro.clone();
	}

	/**
	 * Retorna o formato dos strides (altura, largura) usado pela camada.
	 * @return formato dos strides da camada.
	 */
	public int[] formatoStride() {
		verificarConstrucao();
		return _stride.clone();
	}

	@Override
	public Tensor gradEntrada() {
		verificarConstrucao();
		return _gradEntrada;
	}

	@Override
	public String info() {
		verificarConstrucao();

		StringBuilder sb = new StringBuilder();
		String pad = " ".repeat(4);
		
		sb.append(nome() + " (id " + this.id + ") = [\n");

		sb.append(pad).append("Entrada: " + JNNutils.arrayStr(shapeIn()) + "\n");
		sb.append(pad).append("Filtro: " + JNNutils.arrayStr(_filtro) + "\n");
		sb.append(pad).append("Strides: " + JNNutils.arrayStr(_stride) + "\n");
		sb.append(pad).append("Saída: " + JNNutils.arrayStr(shapeOut()) + "\n");

		sb.append("]\n");

		return sb.toString();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(info());
		int tamanho = sb.length();

		sb.delete(tamanho-1, tamanho);//remover ultimo "\n"    
		
		sb.append(" <hash: " + Integer.toHexString(hashCode()) + ">");
		sb.append("\n");
		
		return sb.toString();
	}

	@Override
	public Pool2DBase clone() {
		Pool2DBase clone = (Pool2DBase) super.clone();

		clone.lops = new LayerOps();

		clone._treinavel = this._treinavel;
		clone.treinando = this.treinando;
		clone._construida = this._construida;

		clone.shapeIn = this.shapeIn.clone();
		clone._filtro = this._filtro.clone();
		clone.shapeOut = this.shapeOut.clone();
		clone._stride = this._stride.clone();
		
		clone._saida = this._saida.clone();
		clone._gradEntrada = this._gradEntrada.clone();

		return clone;
	}
    
	@Override
	public long tamBytes() {
		long tamVars = super.tamBytes(); //base camada + tensores
		tamVars += 4 * shapeIn.length; 
		tamVars += 4 * shapeOut.length; 
		tamVars += 4; //tamLote; 

		long tamTensores =
		_saida.tamBytes() +
		_gradEntrada.tamBytes();

		return tamVars + tamTensores;
	}

}
