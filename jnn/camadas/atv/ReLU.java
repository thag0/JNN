package jnn.camadas.atv;

import jnn.camadas.Camada;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;

public class ReLU extends Camada implements Cloneable {

    Tensor _entrada;
    Tensor _saida;
    Tensor _gradEntrada;
    Tensor _gradSaida;
    Utils utils = new Utils();

    private int[] shapeEntrada;

    public ReLU() {}

    public ReLU(int... shape) {
        construir(shape);
    }

    @Override
    public void construir(int[] shape) {
		utils.validarNaoNulo(shape, "Formato de entrada nulo.");

		if (shape.length < 1) {
			throw new IllegalArgumentException(
				"\nO formato deve conter pelo menos um elemento."
			);
		}

		if (!utils.apenasMaiorZero(shape)) {
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
            (grad, e) -> grad * ((e > 0.0) ? 1.0 : 0.0)
        );

        return _gradEntrada;
    }

    @Override
    public Tensor saida() {
        verificarConstrucao();
        return _saida;
    }

    @Override
    public int[] shapeEntrada() {
        return shapeEntrada;
    }

    @Override
    public int[] shapeSaida() {
        return shapeEntrada();
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
