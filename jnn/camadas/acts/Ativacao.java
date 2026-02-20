package jnn.camadas.acts;

import jnn.camadas.Camada;
import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Camada de ativação base.
 */
public abstract class Ativacao extends Camada implements Cloneable {
    
    /**
     * Tensor contendo os dados de entrada da camada.
     */
    protected Tensor _entrada;

    /**
     * Tensor contendo os dados de saída calculados pela camada.
     */
    protected Tensor _saida;

    /**
     * Tensor contendo os gradientes em relação a entrada da camada.
     */
    protected Tensor _gradEntrada;
    
    /**
     * Tensor contendo os gradientes em relação a saída da camada.
     */
    protected Tensor _gradSaida;

    /**
     * Formato de entrada (e saída) da camada.
     */
    protected int[] shapeIn;

    /**
     * Tamanho do lote de dados usado.
     */
    protected int _tamLote;

    /**
     * Dimensão base da camada usada quando foi construída.
     */
    protected int dimBase;

    /**
     * Construtor privado.
     */
    protected Ativacao() {}

    @Override
    public void construir(int[] shape) {
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
        dimBase = shape.length;

        _saida       = addParam("Saida", shapeIn);
        _gradEntrada = addParam("Grad Entrada", shapeIn);
        _gradSaida   = addParam("Grad Saida", shapeIn);

        _construida = true;
    }

    @Override
    public void inicializar() {}

	@Override
	public void ajustarParaLote(int tamLote) {
		if (tamLote == 0) {
			_gradEntrada = addParam("Grad Entrada", shapeIn);
			_saida = addParam("Saida", shapeIn);
		
		} else {
            int[] shape = new int[shapeIn.length + 1];
            shape[0] = tamLote;
            for (int i = 0; i < shapeIn.length; i++) {
                shape[i+1] = shapeIn[i];
            }

			_gradEntrada = addParam("Grad Entrada", shape);
			_saida = addParam("Saida", shape);
		}
		
		this._tamLote = tamLote;
	}

    /**
     * Pode reajustar os parâmetros da camada caso suas dimensões sejam compatíveis.
     * @param x {@code Tensor} base de entrada.
     */
    protected void talvesAjustarLote(Tensor x) {
        int numDims = x.numDim();

        if (numDims == dimBase) {
            if (_tamLote != 0) {
                ajustarParaLote(0);
            }

        } else if (numDims == dimBase + 1) {
            int lote = x.tamDim(0);

            if (lote != this._tamLote) {
                ajustarParaLote(lote);
            }

        } else {
            throw new UnsupportedOperationException(
                "Esperado tensor com " + dimBase +
                " ou " + (dimBase + 1) +
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
    public Tensor saida() {
        verificarConstrucao();
        return _saida;
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
    public int tamSaida() {
        return saida().tam();
    }

    @Override
    public Tensor gradEntrada() {
        return _gradEntrada;
    }

    @Override
    public Ativacao clone() {
        Ativacao clone = (Ativacao) super.clone();

        clone._saida = _saida.clone();
        clone._gradEntrada = _gradEntrada.clone();
        clone.shapeIn = shapeIn.clone();
    
        return clone;
    }

    @Override
    public long tamBytes() {
        long tamVars = super.tamBytes();//base
        tamVars += 4 * shapeIn.length;
        tamVars += 4;//tamLote
        tamVars += 4;//dimBase
        
        long tamTensores =
        _gradEntrada.tamBytes() +
        _saida.tamBytes();

        return tamVars + tamTensores;
    }

}
