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
	Random random = new Random();//implementar configuração de seed

	/**
	 * Utilitário.
	 */
	Utils utils = new Utils();

	/**
	 * Instancia uma nova camada de dropout, definindo a taxa
	 * de abandono que será usada durante o processo de treinamento.
	 * @param taxa taxa de dropout, um valor entre 0 e 1 representando a
	 * taxa de abandono da camada.
	 */
	public Dropout(double taxa) {
		if (taxa <= 0 || taxa >= 1) {
			throw new IllegalArgumentException(
				"\nTaxa de dropout deve estar entre 0 e 1, " + 
				"recebido: " + taxa
			);
		}

		this.taxa = taxa;
	}

	/**
	 * Instancia uma nova camada de dropout, definindo a taxa
	 * de abandono que será usada durante o processo de treinamento.
	 * @param taxa taxa de dropout, um valor entre 0 e 1 representando a
	 * taxa de abandono da camada.
	 * @param seed seed usada para o gerador de números aleatórios da camada.
	 */
	public Dropout(double taxa, long seed) {
		this(taxa);
		random.setSeed(seed);
	}

	@Override
	public void construir(Object entrada) {
		if (!(entrada instanceof int[])) {
			throw new IllegalArgumentException(
				"\nObjeto esperado para entrada da camada Dropout é do tipo int[], " +
				"objeto recebido é do tipo " + entrada.getClass().getTypeName()
			);
		}

		int[] formato = (int[]) entrada;
		if (!utils.apenasMaiorZero(formato)) {
			throw new IllegalArgumentException(
				"\nOs argumentos do formato de entrada devem ser maiores que zero."
			);
		}
		if (formato.length < 1) {
			throw new IllegalArgumentException(
				"\nO formato deve conter pelo menos um elemento."
			);
		}

		this.shapeEntrada = formato.clone();

		_entrada =     new Tensor(this.shapeEntrada);
		_mascara =     new Tensor(_entrada.shape());
		_saida =       new Tensor(_entrada.shape());
		_gradEntrada = new Tensor(_entrada.shape());

		setNomes();
		
		_construida = true;// camada pode ser usada
	}

	@Override
	public void inicializar() {}

	/**
	 * Configura uma seed fixa para geradores de números aleatórios da
	 * camada.
	 * @param seed nova seed.
	 */
	public void setSeed(long seed) {
		random.setSeed(seed);
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
	 * <h3>
	 *    Nota
	 * </h3>
	 * <p>
	 *    Caso a entrada recebida seja um {@code Tensor}, os valores de entrada
	 *    que serão considerados serão apenas o da primeira dimensão do tensor.
	 * </p>
	 * @param entrada dados de entrada para a camada, objetos aceitos são {@code Tensor}
	 */
	@Override
	public Tensor forward(Object entrada) {
		verificarConstrucao();

		if (entrada instanceof Tensor) {
			Tensor e = (Tensor) entrada;
			if (!_entrada.compararShape(e)) {
				throw new IllegalArgumentException(
					"\nTensor de entrada deve tem formato" + e.shapeStr() + 
					", esperado formato " + _entrada.shapeStr() + "."
				);
			}

			_entrada.copiar(e);
	
		}  else {
			throw new IllegalArgumentException(
				"\nEntrada aceita para a camada de Dropout deve ser do tipo " +
				_entrada.getClass().getSimpleName() + 
				" , objeto recebido é do tipo \"" + entrada.getClass().getTypeName() + "\"."
			);
		}

		_saida.copiar(_entrada);

		if (treinando) {
			gerarMascaras();
			_saida.mult(_mascara);
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
			x ->  (random.nextDouble() >= taxa) ? (1 / (1 - taxa)) : 0.0d
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
	 * @param grad gradientes da camada seguiente.
	 */
	@Override
	public Tensor backward(Object grad) {
		verificarConstrucao();

		if (grad instanceof Tensor) {
			Tensor g = (Tensor) grad;
			if (!_entrada.compararShape(g)) {
				throw new IllegalArgumentException(
					"\nGradiente recebido tem formato" + g.shapeStr() + 
					", esperado formato " + _gradEntrada.shapeStr() + "."
				);
			}

			_gradEntrada.copiar(g);

		} else {
			throw new IllegalArgumentException(
				"\nGradiente aceito para a camada de Dropout deve ser do tipo " + 
				this._gradEntrada.getClass().getTypeName() +
				" ,objeto recebido é do tipo \"" + grad.getClass().getTypeName() + "\"."
			);
		}

		if (treinando) _gradEntrada.mult(_mascara);

		return _gradEntrada;
	}

	@Override
	public Tensor saida() {
		verificarConstrucao();
		return _saida;
	}

	@Override
	public int[] formatoEntrada() {
		verificarConstrucao();
		return shapeEntrada.clone();
	}

	@Override
	public int[] formatoSaida() {
		return formatoEntrada();
	}

	@Override
	public int numParametros() {
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
