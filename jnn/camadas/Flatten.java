package jnn.camadas;

import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;

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
public class Flatten extends Camada implements Cloneable {

	/**
	 * Array contendo o formato de entrada da camada, de acordo com o formato:
	 * <pre>
	 *    entrada = (..., canais, altura, largura)
	 * </pre>
	 */
	private int[] shapeIn;

	/**
	 * Array contendo o formato de saida da camada, de acordo com o formato:
	 * <pre>
	 *    saida = (elementosTotaisEntrada)
	 * </pre>
	 */
	private int[] shapeOut;

	/**
	 * Quatidade de elementos computados ao final do achatamento.
	 */
	private int totalFlatten;

	/**
	 * Auxiliar pra controle de treinamento em lotes.
	 */
	private int tamLote;

	/**
	 * Tensor contendo os valores de entrada para a camada,
	 * que serão usados para o processo de feedforward.
	 * <p>
	 *    O formato da entrada é dado por:
	 * </p>
	 * <pre>
	 *    entrada = (..., canais, altura, largura)
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
	 *    gradEntrada = (..., profundidadeEntrada, alturaEntrada, larguraEntrada)
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
	 *    formEntrada = (canais, altura, largura)
	 * </pre>
	 * @param entrada formato dos dados de entrada para a camada.
	 */
	public Flatten(int[] entrada) {
		construir(entrada);
	}

	/**
	 * Inicializa os parâmetros necessários para a camada Flatten.
	 * <p>
	 *    O formato de entrada deve ser um array contendo o tamanho de 
	 *    cada dimensão de entrada da camada, e deve estar no formato:
	 * </p>
	 * <pre>
	 *    entrada = (canais, altura, largura)
	 * </pre>
	 * Também pode ser aceito um objeto de entrada contendo apenas dois elementos,
	 * eles serão formatados como:
	 * <pre>
	 *    entrada = (altura, largura)
	 * </pre>
	 */
	@Override
	public void construir(int[] shape) {
		JNNutils.validarNaoNulo(shape, "shape == null.");

		if (!JNNutils.apenasMaiorZero(shape)) {
			throw new IllegalArgumentException(
				"\nOs valores do formato de entrada devem ser maiores que zero."
			);
		}

		shapeIn = shape.clone();
		totalFlatten = tamSaida();

		shapeOut = new int[]{ totalFlatten };

		_gradEntrada = addBuffer("Grad Entrada", shapeIn);
		_saida 		 = addBuffer("Saida", shapeOut);

		_construida = true;// camada pode ser usada.
	}

	@Override
	public void inicializar() {}

	@Override
	public void ajustarParaLote(int tamLote) {
		if (tamLote == 0) {
			_gradEntrada = addBuffer("Grad Entrada", shapeIn);
			_saida 		= addBuffer("Saida", tamSaida());
		
		} else {
			int[] shape = new int[shapeIn.length + 1];
			shape[0] = tamLote;
			for (int i = 0; i < shapeIn.length; i++) {
				shape[i+1] = shapeIn[i];
			}

			_gradEntrada = addBuffer("Grad Entrada", shape);
			_saida		= addBuffer("Saida", tamLote, totalFlatten);
		}

		
		this.tamLote = tamLote;
	}

	@Override
	public Tensor forward(Tensor x) {
		verificarConstrucao();

		if (x.numDim() == shapeIn.length) {
			ajustarParaLote(0);
		
		} else if (x.numDim() == shapeIn.length + 1) {
			int lotes = x.tamDim(0);
			if (lotes != this.tamLote) {
				ajustarParaLote(lotes);
			}
		}
		else {
			throw new UnsupportedOperationException(
				"Esperado tensor " + shapeIn.length + "D ou " +
				(shapeIn.length + 1) + "D, mas recebido: " + x.numDim() + "D."
			);
		}

		_entrada = x.contiguous();
		_saida.copiarElementos(_entrada);

		return _saida;
	}

	@Override
	public Tensor backward(Tensor g) {
		verificarConstrucao();

		_gradEntrada.copiarElementos(g);

		return _gradEntrada;
	}

	@Override
	public Tensor saida() {
		verificarConstrucao();
		return _saida;
	}

	@Override
	public float[] saidaParaArray() {
		return saida().array();
	}

	@Override
	public int tamSaida() {
		// calculo individual baseado na entrada pra 
		// evitar problemas com lotes

		int tam = 1; 
		for (int val : shapeIn) {
			tam *= val;
		}

		return tam;
	}

	@Override
	public int[] shapeIn() {
		verificarConstrucao();
		return shapeIn.clone();
	}

	/**
	 * Calcula o formato de saída da camada Flatten, que é disposto da
	 * seguinte forma:
	 * <pre>
	 *    formato = (elementosEntrada)
	 * </pre>
	 * Onde {@code elementosEntrada} é a quantidade total de elementos 
	 * contidos no formato de entrada da camada.
	 * @return formato de saída da camada
	 */
	 @Override
	public int[] shapeOut() {
		verificarConstrucao();
		return _saida.shape().clone();
	}

	@Override
	public int numParams() {
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

		sb.append(pad).append("Entrada: " + JNNutils.arrayStr(shapeIn) + "\n");
		sb.append(pad).append("Saída: (1, " + tamSaida() + ")\n");

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

		clone.shapeIn = shapeIn.clone();
		clone.shapeOut = shapeOut.clone();

		clone._gradEntrada = _gradEntrada.clone();
		clone._saida = _saida.clone();

		return clone;
	}

	@Override
	public long tamBytes() {
		long tamVars = super.tamBytes(); //base camada + tensores
		tamVars += 4 * shapeIn.length; 
		tamVars += 4 * shapeOut.length; 
		tamVars += 4; //totalFlatten
		tamVars += 4; //tamLote

		long tamTensores =
		_saida.tamBytes() +
		_gradEntrada.tamBytes();

		return tamVars + tamTensores;
	}

}
