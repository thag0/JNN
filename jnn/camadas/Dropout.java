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
public class Dropout extends Camada implements Cloneable {

	/**
	 * Taxa de abandono usada durante o treinamento.
	 */
	private float taxa;

	/**
	 * Formato de entrada da camada.
	 */
	private int[] shapeIn;

	/**
	 * Controle da dimensão base da camada.
	 */
	private int dimBase;

	/**
	 * Auxilar no controle de treinamento em lotes.
	 */
	private int tamLote;

	/**
	 * Tensor contendo os valores de entrada para a camada.
	 */
	public Tensor _entrada;

	/**
	 * Tensor contendo as máscaras que serão usadas durante
	 * o processo de treinamento.
	 * <p>
	 *    O formato das máscaras é dado pelo mesmo formato de entrada:
	 * </p>
	 */
	public Tensor _mascara;

	/**
	 * Tensor contendo os valores de saída da camada.
	 * <p>
	 *    O formato de saída é dado pelo mesmo formato de entrada:
	 * </p>
	 */
	public Tensor _saida;

	/**
	 * Tensor contendo os valores dos gradientes que
	 * serão retropropagados durante o processo de treinamento.
	 * <p>
	 *    O formato dos gradientes é dado pelo mesmo formato de entrada:
	 * </p>
	 */
	public Tensor _gradEntrada;

	/**
	 * Instancia uma nova camada de dropout, definindo a taxa
	 * de abandono que será usada durante o processo de treinamento.
	 * @param entrada formato de entrada da camada.
	 * @param taxa taxa de dropout, um valor entre 0 e 1 representando a
	 * taxa de abandono da camada.
	 */
	public Dropout(int[] entrada, Number taxa) {
		this(taxa);
		construir(entrada);
	}

	/**
	 * Instancia uma nova camada de dropout, definindo a taxa
	 * de abandono que será usada durante o processo de treinamento.
	 * @param taxa taxa de dropout, um {@code valor entre 0 e 1} representando a
	 * taxa de abandono da camada.
	 */
	public Dropout(Number taxa) {
		float t = taxa.floatValue();
		
		if (t < 0 || t >= 1) {
			throw new IllegalArgumentException(
				"\nTaxa de dropout deve estar entre 0 e 1, " + 
				"recebido: " + taxa
			);
		}

		this.taxa = t;
	}

	@Override
	public void construir(int[] shape) {
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

		_gradEntrada = addBuffer("Grad Entrada", shapeIn);
		_mascara 	 = addBuffer("Mascara", shapeIn);
		_saida 		 = addBuffer("Saida", shapeIn);

		dimBase = _gradEntrada.numDim();
		
		_construida = true;// camada pode ser usada
	}

	@Override
	public void inicializar() {}

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

		_gradEntrada = addBuffer("Grad Entrada", shape);
		_saida = addBuffer("Saida", _gradEntrada.shape());
		_mascara = addBuffer("Mascara", _gradEntrada.shape());

		this.tamLote = tamLote;
	}

	@Override
	public Tensor forward(Tensor x) {
		verificarConstrucao();

		final int numDim = x.numDim();

		if (numDim == dimBase) {
			validarShapes(x.shape(), shapeIn);
			if (this.tamLote != 0) ajustarParaLote(0);
		
		} else if (numDim == dimBase + 1) {
			validarShapes(x.shape(), shapeIn);
			int lotes = x.tamDim(0);
			if (lotes != this.tamLote) ajustarParaLote(lotes);
		
		} else {
			throw new UnsupportedOperationException(
				"Esperado tensor com " + dimBase +
				" ou " + (dimBase + 1) +
				" dimensões. Recebido: " + x.numDim()
			);
		}

		_entrada = x.contiguous();

		_saida.copiar(_entrada);

		if (treinando) {
			gerarMascaras();
			_saida.mul(_mascara);
		}

		return _saida;
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
	private void gerarMascaras() {
		_mascara.aplicar(
			_ ->  (JNNutils.randFloat() >= taxa) ? (1.0f / (1.0f - taxa)) : 0.0f
		);
	}

	@Override
	public Tensor backward(Tensor g) {
		verificarConstrucao();

		_gradEntrada.copiar(g);

		if (treinando) _gradEntrada.mul(_mascara);

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
	public float taxa() {
		return taxa;
	}

	@Override
	public Dropout clone() {
		verificarConstrucao();

		Dropout clone = (Dropout) super.clone();
		clone.shapeIn = this.shapeIn.clone();
		clone.taxa = this.taxa;

		clone._mascara = this._mascara.clone();
		clone._saida = this._saida.clone();
		clone._gradEntrada = this._gradEntrada.clone();

		return clone;
	}

	@Override
	public String info() {
		verificarConstrucao();

		StringBuilder sb = new StringBuilder();
		String pad = " ".repeat(4);
		
		sb.append(nome() + " (id " + this.id + ") = [\n");

		sb.append(pad).append("Taxa: " + taxa() + "\n");
		sb.append(pad).append("Entrada: " + JNNutils.arrayStr(shapeIn) + "\n");
		sb.append(pad).append("Saída: " + JNNutils.arrayStr(shapeIn) + "\n");

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
	public Tensor gradEntrada() {
		verificarConstrucao();
		return _gradEntrada;
	}

	@Override
	public long tamBytes() {
		long tamVars = super.tamBytes(); //base camada + tensores
		tamVars += 4; //taxa 
		tamVars += 4 * shapeIn.length; 
		tamVars += 4; //dimbase
		tamVars += 4; //tamLote

		long tamTensores = 
		_gradEntrada.tamBytes() +
		_mascara.tamBytes() +
		_saida.tamBytes();

		return tamVars + tamTensores;
	}
}
