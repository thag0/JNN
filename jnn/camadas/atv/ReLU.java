package jnn.camadas.atv;

import jnn.camadas.Camada;
import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * Camada de ativação ReLU.
 */
public class ReLU extends Camada implements Cloneable {

    Tensor _entrada;
    Tensor _saida;
    Tensor _gradEntrada;
    Tensor _gradSaida;

    private int[] shapeEntrada;

    /**
     * Inicializa uma camada de ativação ReLU.
     */
    public ReLU() {}

    /**
     * Inicializa uma camada de ativação ReLU.
     * @param shape formato de entrada para a camada.
     */
    public ReLU(int... shape) {
        construir(shape);
    }

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

        shapeEntrada = shape.clone();

        _entrada     = addParam("Entrada", shapeEntrada);
        _saida       = addParam("Saida", shapeEntrada);
        _gradEntrada = addParam("Grad Entrada", shapeEntrada);
        _gradSaida   = addParam("Grad Saida", shapeEntrada);

        _construida = true;
    }

    @Override
    public void inicializar() {}

    @Override
    public Tensor forward(Tensor x) {
        verificarConstrucao();

		verificarConstrucao();

		_entrada.copiar(x);

        _saida.aplicar(_entrada, v -> v > 0 ? v : 0);

        return _saida;
    }

    @Override
    public Tensor backward(Tensor g) {
		verificarConstrucao();

		_gradSaida.copiar(g);

        _gradEntrada.aplicar(
            _gradSaida, _entrada,
            (grad, e) -> grad * ((e > 0.0) ? 1.0f : 0.0f)
        );

        return _gradEntrada;
    }

    @Override
    public Tensor saida() {
        verificarConstrucao();
        return _saida;
    }

    @Override
    public int[] shapeIn() {
        return shapeEntrada;
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
    public ReLU clone() {
        ReLU clone = (ReLU) super.clone();
    
        clone._entrada = _entrada.clone();
        clone._saida = _saida.clone();
        clone._gradEntrada = _gradEntrada.clone();
        clone._gradSaida = _gradSaida.clone();

        return clone;
    }

    @Override
    public int tamSaida() {
        return saida().tam();
    }

}
