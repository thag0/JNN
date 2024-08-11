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

        _entrada     = new Tensor(shapeEntrada);
        _saida       = new Tensor(shapeEntrada);
        _gradEntrada = new Tensor(shapeEntrada);
        _gradSaida   = new Tensor(shapeEntrada);

        setNomes();

        _construida = true;
    }

	@Override
	protected void setNomes() {
		_entrada.nome("entrada");
		_saida.nome("saida");
		_gradEntrada.nome("grad entrada");    
		_gradSaida.nome("grad saida");
	}

    @Override
    public void inicializar() {}

    @Override
    public Tensor forward(Object entrada) {
        verificarConstrucao();

		verificarConstrucao();

		if (entrada instanceof Tensor) {
			_entrada.copiar((Tensor) entrada);
			
		} else {
			throw new IllegalArgumentException(
				"\nTipo de entrada \"" + entrada.getClass().getTypeName() + "\"" +
				" não suportada."
			);
		}

        _saida.aplicar(_entrada, x -> x > 0 ? x : 0);

        return _saida;
    }

    @Override
    public Tensor backward(Object grad) {
		verificarConstrucao();

		if (grad instanceof Tensor) {
			_gradSaida.copiar((Tensor) grad);
			
		} else {
			throw new IllegalArgumentException(
				"\nTipo de gradiente \"" + grad.getClass().getTypeName() + "\"" +
				" não suportado."
			);
		}

        _gradEntrada.aplicar(
            _gradSaida, _entrada,
            (g, e) -> g * ((e > 0.0) ? 1.0 : 0.0)
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
    public void copiarParaTreinoLote(Camada camada) {
        if (!(camada instanceof ReLU)) {
			throw new UnsupportedOperationException(
				"\nCamada deve ser do tipo " + getClass() +
				", mas é do tipo " + camada.getClass()
			);
        }

		ReLU c = (ReLU) camada;
		_entrada.copiar(c._entrada);
		_saida.copiar(c._saida);
    }

    @Override
    public int tamSaida() {
        return saida().tam();
    }

}
