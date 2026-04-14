package jnn.camadas;

import jnn.core.JNNutils;
import jnn.core.Parametro;
import jnn.core.tensor.Tensor;

/**
 * <h2>
 *    Camada base
 * </h2>
 * <p>
 *    A classe camada serve de molde para criação de novas camadas e 
 *    não pode ser especificamente instanciada nem utilizada.
 * </p>
 * <p>
 *    {@code Não é recomendado} fazer atribuições ou alterações diretamente 
 *    dos atributos de camadas filhas fora da biblioteca, eles estão publicos
 *    apenas pela facilidade de manuseio. Para estes é recomendado usar
 *    os métodos propostos pelas camadas.
 * </p>
 * <p>
 *    As partes mais importantes de uma camada são {@code forward()} e 
 *    {@code backward()} onde são implementados os métodos básicos para
 *    propagação e retropropagação de dados.
 * </p>
 * <p>
 *    Para a parte de propagação direta (ou forward) os dados recebidos de entrada 
 *    são processados de acordo com cada regra individual de cada camada e ao final 
 *    os resultados são salvos em sua saída.
 * </p>
 * <p>
 *    Na propagação reversa (ou backward) são recebidos os gradientes da camada 
 *    anterior e cada camada irá fazer seu processamento para calcular os próprios 
 *    gradientes para seus atributos treináveis. Aqui cada camada tem o adicional 
 *    de calcular os gradientes em relação as suas entradas para retropropagar para 
 *    camadas anteriores usadas pelos modelos.
 * </p>
 * <h2>
 *    Detalhe adicional.
 * </h2>
 * <ul>
 *	<li>
 *		Os elementos das camadas devem ser pré inicializados para evitar alocações 
 *		dinâmicas durante a execução dos modelos e isso se dá por dois motivos: ter 
 *		controle das dimensões dos objetos criandos durante toda a execução dos 
 *		algoritmos e também criar uma espécie de cache para evitar muitas instanciações
 *		em runtime.
 *	</li>
 * </ul>
 */
public abstract class Camada {

	// TODO fazer validação dos shapes dos tensores recebidos no backward

	/**
	 * Controlador para uso dentro dos algoritmos de treino.
	 */
	protected boolean _treinavel = false;

	/**
	 * Controlador de construção da camada.
	 */
	protected boolean construida = false;

	/**
	 * Controlador de treino da camada.
	 */
	protected boolean treinando = false;

	/**
	 * Auxilar no controle de treinamento em lotes.
	 */
	protected int tamLote;

	/**
	 * Identificador único da camada.
	 */
	public int id;

	/**
	 * Array de parâmetros treináveis.
	 */
	Parametro[] _params = {};

	/**
	 * Instancia a camada base usada dentro dos modelos de Rede Neural.
	 * <p>
	 *    A camada base não possui implementação de métodos e é apenas usada
	 *    como molde de base para as outras camadas terem suas próprias implementações.
	 * </p>
	 */
	protected Camada() {}

	/**
	 * Aloca a estrutura da camada, alocando os recursos necessários 
	 * para funcionamento.
	 * <p>
	 *		Essa operação NÃO incializa valores, apenas aloca em 
	 *		memória os atributos da camada.
	 * </p>
	 * <p>
	 *		Para inicializar kernels e bias use <strong>inicializar()</strong>
	 * </p>
	 * @param shape formato de entrada da camada.
	 */
	public abstract void construir(int[] shape);

	/**
	 * Verificador de inicialização para evitar problemas.
	 */
	protected void verificarConstrucao() {
		if (!construida) {
			throw new IllegalStateException(
				"\nCamada " + nome() + " (id = " + id + ") não foi construída."
			);
		}
	}

	/**
	 * Verifica se o shape recebido é compatível com o shape esperado.
	 * @param base shape recebido.
	 * @param alvo shape esperado.
	 */
	protected void validarShapes(int[] base, int[] alvo) {
		// assumindo que base pode ter uma dimensão a mais para o lote.

		int lb = base.length;
		int la = alvo.length;

		// comparando assim pra ignorar a primeira dimensão (lote)
		for (int i = 1; i <= la; i++) {
			if (base[lb - i] != alvo[la - i]) {
				throw new IllegalArgumentException(
					"\nShapes incompatíveis, últimas dimensões esperadas = " + JNNutils.arrayStr(alvo) +
					", recebido = " + JNNutils.arrayStr(base)
				);
			}
		}
	}

	/**
	 * Gera os valores iniciais para parâmetros treináveis da camada de 
	 * acordo com os seus inicializadores definidos.
	 */
	public abstract void init();

	/**
	 * Configura o id da camada. O id deve indicar dentro de um modelo, em 
	 * qual posição a camada está localizada.
	 * @param id id da camada.
	 */
	public void setId(Number id) {
		int i = id.intValue();
		
		if (i < 0) {
			throw new IllegalArgumentException(
				"\nId da camada deve ser maior ou igual a zero, recebido: " + i + "."
			);
		}

		this.id = i;
	}

	/**
	 * Configura o uso do bias para a camada.
	 * <p>
	 *    A configuração deve ser feita antes da construção da camada.
	 * </p>
	 * @param usarBias uso do bias.
	 */
	public void setBias(boolean usarBias) {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui configuração de bias."
		);    
	}

	/**
	 * Configura a camada para treino.
	 * @param treinando caso verdadeiro a camada será configurada para
	 * treino, caso contrário, será usada para testes/predições.
	 */
	public void setTreino(boolean treinando) {
		this.treinando = treinando;
	}

	/**
	 * Inicializa um {@code Tensor} vazio com nome especificado.
	 * <p>
	 *		O tensor criado é adicionado aos parâmetros treináveis da camada,
	 *		este que é passado para o otimizador para atualização durante o treino.
	 * </p>
	 * @param shape {@code array} contendo os valores das dimensões do tensor.
	 * @param nome {@code String} contendo nome desejado.
	 * @return {@code Tensor} criado.
	 */
	protected void addParam(String nome, int... shape) {
		Parametro p = new Parametro(nome, shape);
		_params = JNNutils.addEmArray(_params, p);
	}

	/**
	 * Inicializa um {@code Tensor} vazio com nome especificado.
	 * @param shape {@code array} contendo os valores das dimensões do tensor.
	 * @param nome {@code String} contendo nome desejado.
	 * @return {@code Tensor} criado.
	 */
	protected Tensor addBuffer(String nome, int... shape) {
		Tensor b = new Tensor(shape).nome(nome);
		// aqui não é adicionado nada
		return b;
	}

	/**
	 * Adapta os parâmetros relevante da camada para lidar
	 * com lotes de dados.
	 * <p>
	 *		Essa operação não altera kernels e bias.
	 * </p>
	 * @param tamLote tamanho do lote de dados.
	 */
	public void ajustarParaLote(int tamLote) {
		throw new UnsupportedOperationException(
			"\nNão implementado."
		);
	}

	/**
	 * Propaga os dados de entrada pela camada.
	 * @param x dados de entrada que serão processados.
	 * @return {@code Tensor} contendo a saída calculada pela camada.
	 */
	public abstract Tensor forward(Tensor x);

	/**
	 * Retropropaga os gradientes recebidos para as camadas anteriores.
	 * @param g gradiente em relação a saída da camada.
	 * @return {@code Tensor} contendo os gradientes em relação a entrada da camada.
	 */
	public abstract Tensor backward(Tensor g);

	/**
	 * Retorna a saída da camada.
	 * @return saída da camada.
	 */
	public abstract Tensor saida();
	
	/**
	 * Retorna o formado dos dados de entrada suportados pela camada.
	 * @return formato de entrada da camada.
	 */
	public abstract int[] shapeIn();

	/**
	 * Retorna o formado dos dos dados de saída gerados pela camada.
	 * @return formato de saída da camada.
	 */
	public abstract int[] shapeOut();

	/**
	 * Retorna a saída da camada no formato de array.
	 * @return saída da camada.
	 */
	public float[] saidaParaArray() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui retorno de saída para array."
		);    
	}

	/**
	 * Retorna a quantidade total de elementos presentes na saída da camada.
	 * @return tamanho de saída da camada.
	 */
	public int tamSaida() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui retorno de tamanho da saída."
		);      
	}

	/**
	 * Retorna a quantidade de parâmetros treináveis da camada.
	 * <p>
	 *    Esses parâmetros podem incluir pesos, filtros, bias, entre outros.
	 * </p>
	 * O resultado deve ser a quantidade total desses elementos.
	 * @return número da parâmetros da camada.
	 */
	public abstract int numParams();

	/**
	 * Retorna o verificador de uso do bias dentro da camada.
	 * @return uso de bias na camada.
	 */
	public boolean temBias() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui verificação de bias."
		);
	}

	/**
	 * Retorna o bias da camada.
	 * <p>
	 *    <strong> O bias só existe em camadas treináveis </strong>.
	 * </p>
	 * @return {@code Tensor} de bias da camada.
	 */
	public Tensor bias() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui bias."
		);      
	}

	/**
	 * Retorna o gradiente em relação à entrada da camada.
	 * @return {@code Tensor} contendo os gradientes em relação à entrada da camada.
	 */
	public Tensor gradEntrada() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui gradiente de entrada."
		);    
	}

	/**
	 * Zera os gradientes de parâmetros treináveis.
	 */
	public void gradZero() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui kernel/bias para zerar."
		);
	}

	/**
	 * Verifica se a camada é treinável.
	 * @return {@code true} caso a camada seja treinável, {@code false}
	 * caso contrário.
	 */
	public boolean treinavel() {
		return _treinavel;
	}

	/**
	 * Clona as características principais da camada.
	 * @return clone da camada.
	 */
	@Override
	public Camada clone() {
		try {
			return (Camada) super.clone();
		} catch (CloneNotSupportedException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Retorna o nome da camada.
	 * @return nome da camada.
	 */
	public String nome() {
		return getClass().getSimpleName();
	}

	/**
	 * Retorna uma {@code String} contendo um resumo das informações
	 * da camada.
	 * @return informações da camada.
	 */
	public String info() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui implementação para info()"
		);
	}

    @Override
    public String toString() {
		StringBuilder sb = new StringBuilder(info());

		int pos = sb.lastIndexOf("]"); // encontra o fechamento

		String hashStr = "    <hash: " + Integer.toHexString(hashCode()) + ">\n";

		sb.insert(pos, hashStr);

		return sb.toString();
    }

	/**
	 * Retorna o tamanho em bytes na memória ocupado pela camada.
	 * @return tamanho aproximado em bytes.
	 */
	public long tamBytes() {
		String jvmBits = System.getProperty("sun.arch.data.model");
        long bits = Long.valueOf(jvmBits);

        long tamObj;
		// overhead da jvm
        if (bits == 32) tamObj = 8;
        else if (bits == 64) tamObj = 16;
        else throw new IllegalStateException(
            "\nSem suporte para plataforma de " + bits + " bits."
        );

		long tamTensores = 0;
		if (treinavel()) {
			for (var param : _params) {
				tamTensores += param.weight.tamBytes();
				tamTensores += param.grad.tamBytes();
			}
		}

		return tamObj +
			1 + //treinavel 
			1 + //construida
			1 + //treinando
			4 + //tamLote
			4 + //id
			tamTensores;
	}

	/**
	 * Retorna um array contendo os parâmetros treináveis da camada.
	 * @return array de parâmetros treináveis da camada.
	 */
	public Parametro[] params() {
		return _params;
	}

	/**
	 * Retorna se a camada está construída.
	 * @return camada construída.
	 */
	public boolean construida() {
		return construida;
	}

}
