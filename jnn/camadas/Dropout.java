package jnn.camadas;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

/**
 * <p>
 *    Camada de Abandono
 * </p>
 * <p>
 *    O dropout (abandono) é uma técnica regularizadora usada para evitar
 *    overfitting para modelos muito grandes, melhorando sua capacidade de 
 *    generalização.
 * </p>
 * <p>
 *    Durante o treinamento, o dropout "desativa" temporariamente e aleatoriamente 
 *    algumas unidades/neurônios da camada, impedindo que eles contribuam para o 
 *    cálculo da saída. Isso força o modelo a aprender a partir de diferentes 
 *    subconjuntos dos dados em cada iteração, o que ajuda a evitar a dependência 
 *    excessiva de determinadas unidades e resulta numa representação mais generalista
 *    do conjunto de dados.
 * </p>
 * <p>
 *    A camada de dropout não possui parâmetros treináveis nem função de ativação.
 * </p>
 */
public class Dropout extends Layer implements Cloneable {

	/**
	 * Taxa de abandono usada durante o treinamento.
	 */
	private float rate;

	/**
	 * Formato de entrada da camada.
	 */
	private int[] shapeIn;

	/**
	 * Controle da dimensão base da camada.
	 */
	private int baseDim;

	/**
	 * Tensor contendo os valores de entrada para a camada.
	 */
	public Tensor _input;

	/**
	 * Tensor contendo as máscaras que serão usadas durante
	 * o processo de treinamento.
	 * <p>
	 *    O formato das máscaras é dado pelo mesmo formato de entrada:
	 * </p>
	 */
	public Tensor _mask;

	/**
	 * Tensor contendo os valores de saída da camada.
	 * <p>
	 *    O formato de saída é dado pelo mesmo formato de entrada:
	 * </p>
	 */
	public Tensor _output;

	/**
	 * Tensor contendo os valores dos gradientes que
	 * serão retropropagados durante o processo de treinamento.
	 * <p>
	 *    O formato dos gradientes é dado pelo mesmo formato de entrada:
	 * </p>
	 */
	public Tensor _gradInput;

	/**
	 * Instancia uma nova camada de dropout, definindo a taxa
	 * de abandono que será usada durante o processo de treinamento.
	 * @param input formato de entrada da camada.
	 * @param rate taxa de dropout, um valor entre 0 e 1 representando a
	 * taxa de abandono da camada.
	 */
	public Dropout(int[] input, Number rate) {
		this(rate);
		build(input);
	}

	/**
	 * Instancia uma nova camada de dropout, definindo a taxa
	 * de abandono que será usada durante o processo de treinamento.
	 * @param rate taxa de dropout, um {@code valor entre 0 e 1} representando a
	 * taxa de abandono da camada.
	 */
	public Dropout(Number rate) {
		float t = rate.floatValue();
		
		if (t < 0 || t >= 1) {
			throw new IllegalArgumentException(
				"\nTaxa de dropout deve estar entre 0 e 1, " + 
				"recebido: " + rate
			);
		}

		this.rate = t;
	}

	@Override
	public void build(int[] shape) {
		JNNutils.validarNaoNulo(shape, "shape == null.");

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

		_gradInput = addBuffer("grad input", shapeIn);
		_mask 	   = addBuffer("mask", shapeIn);
		_output    = addBuffer("grad output", shapeIn);

		baseDim = _gradInput.numDim();
		
		_built = true;// camada pode ser usada
	}

	@Override
	public void initParams() {}

	@Override
	public void fitToBatch(int tamLote) {
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

		_gradInput = addBuffer("grad input", shape);
		_output = addBuffer("output", _gradInput.shape());
		_mask = addBuffer("mask", _gradInput.shape());

		this.batchSize = tamLote;
	}

	@Override
	public Tensor forward(Tensor x) {
		isBuilt();

		final int numDim = x.numDim();

		if (numDim == baseDim) {
			validateShapes(x.shape(), shapeIn);
			if (this.batchSize != 0) fitToBatch(0);
		
		} else if (numDim == baseDim + 1) {
			validateShapes(x.shape(), shapeIn);
			int lotes = x.lenDim(0);
			if (lotes != this.batchSize) fitToBatch(lotes);
		
		} else {
			throw new UnsupportedOperationException(
				"Esperado tensor com " + baseDim +
				" ou " + (baseDim + 1) +
				" dimensões. Recebido: " + x.numDim()
			);
		}

		_input = x.contiguous();

		_output.copy(_input);

		if (_training) {
			generateMasks();
			_output.mul(_mask);
		}

		return _output;
	}

	/**
	 * Gera a máscara aleatória para cada camada de entrada que será 
	 * usada durante o processo de treinamento.
	 * <p>
	 *    Exemplo:
	 * </p>
	 * <pre>
	 *mascara = [
	 *    [[1.0, 0.0, 0.0],  
	 *     [0.0, 1.0, 1.0],  
	 *     [0.0, 1.0, 0.0]]  
	 *]
	 * </pre>
	 * Nos valores em que a máscara for igual a 1, o valor de entrada será
	 * passado para a saída, nos valores iguais a 0, a entrada será desconsiderada. 
	 */
	private void generateMasks() {
		_mask.apply(
			_ ->  (JNNutils.randFloat() >= rate) ? (1.0f / (1.0f - rate)) : 0.0f
		);
	}

	@Override
	public Tensor backward(Tensor g) {
		isBuilt();

		_gradInput.copy(g);

		if (_training) _gradInput.mul(_mask);

		return _gradInput;
	}

	@Override
	public Tensor output() {
		isBuilt();
		return _output;
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
		return 0;
	}

	/**
	 * Retorna a taxa de dropout usada pela camada.
	 * @return taxa de dropout da camada.
	 */
	public float rate() {
		return rate;
	}

	@Override
	public Dropout clone() {
		isBuilt();

		Dropout clone = (Dropout) super.clone();
		clone.shapeIn = this.shapeIn.clone();
		clone.rate = this.rate;

		clone._mask = this._mask.clone();
		clone._output = this._output.clone();
		clone._gradInput = this._gradInput.clone();

		return clone;
	}

	@Override
	public String info() {
		isBuilt();

		StringBuilder sb = new StringBuilder();
		String pad = " ".repeat(4);
		
		sb.append(name()).append("(\n");

		sb.append(pad).append("In: ").append(JNNutils.arrayStr(shapeIn)).append("\n");
		sb.append(pad).append("Out: ").append(JNNutils.arrayStr(shapeIn)).append("\n");
		sb.append(pad).append("Taxa: ").append(rate()).append("\n");

		sb.append(")");

		return sb.toString();
	}

	@Override
	public Tensor gradInput() {
		isBuilt();
		return _gradInput;
	}

	@Override
	public long sizeBytes() {
		long tamVars = super.sizeBytes(); //base camada + tensores
		tamVars += 4; //taxa 
		tamVars += 4 * shapeIn.length; 
		tamVars += 4; //dimbase

		long tamTensores = 
		_gradInput.sizeBytes() +
		_mask.sizeBytes() +
		_output.sizeBytes();

		return tamVars + tamTensores;
	}
}
