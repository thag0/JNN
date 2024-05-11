package jnn.camadas;

import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.Variavel;

/**
 * <h2>
 *    Camada flatten ou "achatadora"
 * </h2>
 * <p>
 *    Transforma os recebidos no formato sequencial.
 * </p>
 * <p>
 *    A camada de flatten não possui parâmetros treináveis nem função de ativação.
 * </p>
 */
public class Flatten extends Camada implements Cloneable{

	/**
	 * Utilitário.
	 */
	Utils utils = new Utils();

	/**
	 * Array contendo o formato de entrada da camada, de acordo com o formato:
	 * <pre>
	 *    entrada = (profundidade, altura, largura)
	 * </pre>
	 */
	int[] formEntrada;

	/**
	 * Array contendo o formato de saida da camada, de acordo com o formato:
	 * <pre>
	 *    saida = (elementosTotaisEntrada)
	 * </pre>
	 */
	int[] formSaida;

	/**
	 * Tensor contendo os valores de entrada para a camada,
	 * que serão usados para o processo de feedforward.
	 * <p>
	 *    O formato da entrada é dado por:
	 * </p>
	 * <pre>
	 *    entrada = (profundidade, altura, largura)
	 * </pre>
	 */
	public Tensor _entrada;
	
	/**
	 * Tensor contendo os valores dos gradientes usados para 
	 * a retropropagação para camadas anteriores.
	 * <p>
	 *    O formato dos gradientes é dado por:
	 * </p>
	 * <pre>
	 *    gradEntrada = (profundidadeEntrada, alturaEntrada, larguraEntrada)
	 * </pre>
	 */
	public Tensor _gradEntrada;

	/**
	 * Tensor contendo a saída achatada da camada.
	 * <p>
	 *    Mesmo a saída sendo um tensor, ela possui apenas
	 *    uma linha e o número de colunas é equivalente a quantidade
	 *    total de elementos da entrada.
	 * </p>
	 */
	public Tensor _saida;

	/**
	 * Inicializa uma camada Flatten, que irá achatar a entrada recebida
	 * no formato de um tensor unidimensional.
	 * <p>
	 *    É necessário construir a camada para que ela possa ser usada.
	 * </p>
	 */
	public Flatten() {}

	/**
	 * Instancia uma camada Flatten, que irá achatar a entrada recebida
	 * no formato de um tensor unidimensional.
	 * <p>
	 *    O formato de entrada da camada deve seguir o formato:
	 * </p>
	 * <pre>
	 *    formEntrada = (profundidade, altura, largura)
	 * </pre>
	 * @param formEntrada formato dos dados de entrada para a camada.
	 */
	public Flatten(int[] formEntrada) {
		construir(formEntrada);
	}

	/**
	 * Inicializa os parâmetros necessários para a camada Flatten.
	 * <p>
	 *    O formato de entrada deve ser um array contendo o tamanho de 
	 *    cada dimensão de entrada da camada, e deve estar no formato:
	 * </p>
	 * <pre>
	 *    entrada = (profundidade, altura, largura)
	 * </pre>
	 * Também pode ser aceito um objeto de entrada contendo apenas dois elementos,
	 * eles serão formatados como:
	 * <pre>
	 *    entrada = ( altura, largura)
	 * </pre>
	 * @param entrada formato de entrada para a camada.
	 */
	@Override
	public void construir(Object entrada) {
		utils.validarNaoNulo(entrada, "\nFormato de entrada fornecida para camada Flatten é nulo.");

		if (!(entrada instanceof int[])) {
			throw new IllegalArgumentException(
				"\nObjeto esperado para entrada da camada Flatten é do tipo int[], " +
				"objeto recebido é do tipo " + entrada.getClass().getTypeName()
			);
		}

		int[] formatoEntrada = (int[]) entrada;
		if (!utils.apenasMaiorZero(formatoEntrada)) {
			throw new IllegalArgumentException(
				"\nOs valores do formato de entrada devem ser maiores que zero."
			);
		}

		int profundidade, altura, largura;
		if (formatoEntrada.length == 4) {
			profundidade = formatoEntrada[1];
			altura = formatoEntrada[2];
			largura = formatoEntrada[3];

		} else if (formatoEntrada.length == 3) {
			profundidade = formatoEntrada[0];
			altura = formatoEntrada[1];
			largura = formatoEntrada[2];
		
		} else if (formatoEntrada.length == 2) {
			profundidade = 1;
			altura = formatoEntrada[0];
			largura = formatoEntrada[1];
		
		} else {
			throw new IllegalArgumentException(
				"O formato de entrada para a camada Flatten deve conter dois " + 
				"elementos (altura, largura), três elementos (profundidade, altura, largura), " +
				" ou quatro elementos (primeiro desconsiderado) " +
				"objeto recebido possui " + formatoEntrada.length + " elementos."
			);
		}

		//inicialização de parâmetros

		this.formEntrada = new int[]{
			profundidade,
			altura,
			largura
		};

		int tamanho = 1;
		for(int i : this.formEntrada) {
			tamanho *= i;
		}

		this.formSaida = new int[]{tamanho};

		_entrada = new Tensor(formEntrada);
		_gradEntrada = new Tensor(_entrada.shape());
		_saida = new Tensor(formSaida);

		setNomes();

		_construida = true;//camada pode ser usada.
	}

	@Override
	public void inicializar() {}

	@Override
	public void setSeed(long seed) {}

	@Override
	protected void setNomes() {
		_entrada.nome("entrada");
		_saida.nome("saída");
		_gradEntrada.nome("gradiente entrada");     
	}

	/**
	 * <h2>
	 *    Propagação direta através da camada Flatten
	 * </h2>
	 * Achata os dados de entrada num formato sequencial.
	 * @param entrada dados de entrada que serão processados, objetos aceitos incluem:
	 * {@code Tensor}
	 */
	@Override
	public Tensor forward(Object entrada) {
		verificarConstrucao();

		if (entrada instanceof Tensor) {
			Tensor e = (Tensor) entrada;
			if (!_entrada.compararShape(e)) {
				throw new IllegalArgumentException(
					"\nDimensões da entrada recebida " + e.shapeStr() +
					" incompatíveis com a entrada da camada " + _entrada.shapeStr()
				);
			}

			_entrada.copiar(e);

		} else {
			throw new IllegalArgumentException(
				"A camada Flatten não suporta entradas do tipo \"" + entrada.getClass().getTypeName() + "\"."
			);
		}

		_saida.copiarElementos(_entrada.dados);

		return _saida;
	}

	/**
	 * <h2>
	 *    Propagação reversa através da camada Flatten
	 * </h2>
	 * Desserializa os gradientes recebedos de volta para o mesmo formato de entrada.
	 * @param grad gradientes de entrada da camada seguinte, objetos aceitos incluem:
	 * {@code Tensor}.
	 */
	@Override
	public Tensor backward(Object grad) {
		verificarConstrucao();

		if (grad instanceof Tensor) {
			Tensor g = (Tensor) grad;
			if (_gradEntrada.compararShape(g)) {
				throw new IllegalArgumentException(
					"\nDimensões do gradiente recebido " + g.shapeStr() +
					"inconpatíveis com o suportado pela camada " + _gradEntrada.shapeStr()
				);
			}

			_gradEntrada.copiarElementos(g.dados);
		
		} else {
			throw new IllegalArgumentException(
				"A camada Flatten não suporta gradientes do tipo \"" + grad.getClass().getTypeName() + "\"."
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
	public Variavel[] saidaParaArray() {
		verificarConstrucao();
		return saida().paraArray();
	}

	@Override
	public int tamanhoSaida() {
		verificarConstrucao();
		return saida().tamanho();
	}

	@Override
	public int[] formatoEntrada() {
		verificarConstrucao();
		return formEntrada.clone();
	}

	/**
	 * Calcula o formato de saída da camada Flatten, que é disposto da
	 * seguinte forma:
	 * <pre>
	 *    formato = (1, 1, 1, elementosEntrada)
	 * </pre>
	 * Onde {@code elementosEntrada} é a quantidade total de elementos 
	 * contidos no formato de entrada da camada.
	 * @return formato de saída da camada
	 */
	 @Override
	public int[] formatoSaida() {
		verificarConstrucao();
		
		return new int[]{
			1,
			tamanhoSaida()
		};
	}

	@Override
	public int numParametros() {
		return 0;
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
		
		sb.append(nome() + " (id " + id + ") = [\n");

		sb.append(pad).append("Entrada: " + utils.shapeStr(formEntrada) + "\n");
		sb.append(pad).append("Saída: (1, " + tamanhoSaida() + ")\n");

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
	public Flatten clone() {
		Flatten clone = (Flatten) super.clone();

		clone._treinavel = this._treinavel;
		clone.treinando = this.treinando;
		clone._construida = this._construida;

		clone.formEntrada = formEntrada.clone();
		clone.formSaida = formSaida.clone();

		clone._entrada = _entrada.clone();
		clone._gradEntrada = _gradEntrada.clone();
		clone._saida = _saida.clone();

		return clone;
	}

}
