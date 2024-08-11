package jnn.camadas;

import java.util.Random;

import jnn.core.Utils;
import jnn.core.tensor.Tensor;

/**
 * <h1>
 *    Camada de Abandono
 * </h1>
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
	private double taxa;

	/**
	 * Formato de entrada da camada.
	 */
	int[] shapeEntrada;

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
	 * Gerador de valores aleatórios.
	 */
	Random random = new Random();// implementar configuração de seed

	/**
	 * Utilitário.
	 */
	Utils utils = new Utils();

	/**
	 * Instancia uma nova camada de dropout, definindo a taxa
	 * de abandono que será usada durante o processo de treinamento.
	 * @param entrada formato de entrada da camada.
	 * @param taxa taxa de dropout, um valor entre 0 e 1 representando a
	 * taxa de abandono da camada.
	 * @param seed seed usada para o gerador de números aleatórios da camada.
	 */
	public Dropout(int[] entrada, Number taxa, Number seed) {
		this(taxa, seed);
		construir(entrada);
	}

	/**
	 * Instancia uma nova camada de dropout, definindo a taxa
	 * de abandono que será usada durante o processo de treinamento.
	 * @param taxa taxa de dropout, um valor entre 0 e 1 representando a
	 * taxa de abandono da camada.
	 * @param seed seed usada para o gerador de números aleatórios da camada.
	 */
	public Dropout(Number taxa, Number seed) {
		this(taxa);
		setSeed(seed);
	}

	/**
	 * Instancia uma nova camada de dropout, definindo a taxa
	 * de abandono que será usada durante o processo de treinamento.
	 * @param taxa taxa de dropout, um {@code valor entre 0 e 1} representando a
	 * taxa de abandono da camada.
	 */
	public Dropout(Number taxa) {
		double t = taxa.doubleValue();
		
		if (t <= 0 || t >= 1) {
			throw new IllegalArgumentException(
				"\nTaxa de dropout deve estar entre 0 e 1, " + 
				"recebido: " + taxa
			);
		}

		this.taxa = t;
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

		_entrada =     new Tensor(shapeEntrada);
		_mascara =     new Tensor(_entrada.shape());
		_saida =       new Tensor(_entrada.shape());
		_gradEntrada = new Tensor(_entrada.shape());

		setNomes();
		
		_construida = true;// camada pode ser usada
	}

	@Override
	public void inicializar() {}

	@Override
	public void setSeed(Number seed) {
		if (seed != null) {
			random.setSeed(seed.longValue());
		}
	}

	@Override
	protected void setNomes() {
		_entrada.nome("entrada");
		_mascara.nome("máscara");
		_saida.nome("saida");
		_gradEntrada.nome("gradiente entrada");    
	}

	/**
	 * <h2>
	 *    Propagação direta através da camada Dropout
	 * </h2>
	 * <p>
	 *    A máscara de dropout é gerada e utilizada apenas durante o processo
	 *    de treinamento, durante isso, cada predição irá gerar uma máscara diferente,
	 *    a máscara será usada como filtro para os dados de entrada e os valores de 
	 *    unidades desativadas serão propagados como "0".
	 * </p>
	 * <p>
	 *    Caso a camada não esteja {@code treinando}, por padrão a entrada é apenas
	 *    repassada para a saída da camada.
	 * </p>
	 */
	@Override
	public Tensor forward(Object x) {
		verificarConstrucao();

		_entrada.copiar(utils.paraTensor(x));

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
			x ->  (random.nextDouble() >= taxa) ? (1 / (1 - taxa)) : 0.0
		);
	}

	/**
	 * <h2>
	 *    Propagação reversa através da camada Dropout
	 * </h2>
	 * Retropropaga os gradientes da camada suguinte, aplicando a máscara usada
	 * no cálculo da saída.
	 * <p>
	 *    Caso a camada não esteja {@code treinando}, por padrão os gradientes são 
	 *    apenas repassados para o gradiente de entrada da camada.
	 * </p>
	 */
	@Override
	public Tensor backward(Object grad) {
		verificarConstrucao();

		_gradEntrada.copiar(utils.paraTensor(grad));

		if (treinando) _gradEntrada.mul(_mascara);

		return _gradEntrada;
	}

	@Override
	public Tensor saida() {
		verificarConstrucao();
		return _saida;
	}

	@Override
	public int[] shapeEntrada() {
		verificarConstrucao();
		return shapeEntrada.clone();
	}

	@Override
	public int[] shapeSaida() {
		return shapeEntrada();
	}

	@Override
	public int numParams() {
		return 0;
	}

	/**
	 * Retorna a taxa de dropout usada pela camada.
	 * @return taxa de dropout da camada.
	 */
	public double taxa() {
		return this.taxa;
	}

	@Override
	public Dropout clone() {
		verificarConstrucao();

		Dropout clone = (Dropout) super.clone();
		clone.shapeEntrada = this.shapeEntrada.clone();
		clone.taxa = this.taxa;
		clone.random = new Random();

		clone._entrada = this._entrada.clone();
		clone._mascara = this._mascara.clone();
		clone._saida = this._saida.clone();
		clone._gradEntrada = this._gradEntrada.clone();

		return clone;
	}

	@Override
	public void copiarParaTreinoLote(Camada camada) {
		if (!(camada instanceof Dropout)) {
			throw new UnsupportedOperationException(
				"\nCamada deve ser do tipo " + getClass() +
				", mas é do tipo " + camada.getClass()
			);
		}

		Dropout c = (Dropout) camada;
		_entrada.copiar(c._entrada);
		_mascara.copiar(c._mascara);// necessário
		_saida.copiar(c._saida);
	}

	@Override
	public String info() {
		verificarConstrucao();

		StringBuilder sb = new StringBuilder();
		String pad = " ".repeat(4);
		
		sb.append(nome() + " (id " + this.id + ") = [\n");

		sb.append(pad).append("Taxa: " + taxa() + "\n");
		sb.append(pad).append("Entrada: " + utils.shapeStr(shapeEntrada) + "\n");
		sb.append(pad).append("Saída: " + utils.shapeStr(shapeEntrada) + "\n");

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
}
