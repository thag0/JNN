package jnn.camadas;

import jnn.ativacoes.Ativacao;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.Variavel;

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
 *    Existem dois detalhes importantes na implementação das camadas.
 * </h2>
 * <ul>
 *    <li>
 *       Primeiramente que os elementos das camadas devem ser pré inicializados 
 *       para evitar alocações dinâmicas durante a execução dos modelos e isso 
 *       se dá por dois motivos: ter controle das dimensões dos objetos criandos 
 *       durante toda a execução dos algoritmos e também criar uma espécie de cache 
 *       para evitar muitas instanciações em runtime.
 *    </li>
 *    <li>
 *       Segundo, que as funções de ativação não são camadas independentes e sim 
 *       funções que atuam sobre os elementos das camadas, especialmente nos elementos 
 *       chamados "somatório" e guardam os resultados na saída da camada.
 *    </li>
 * </ul>
 */
public abstract class Camada {

	/**
	 * Controlador para uso dentro dos algoritmos de treino.
	 */
	protected boolean _treinavel = false;

	/**
	 * Controlador de construção da camada.
	 */
	public boolean _construida = false;

	/**
	 * Controlador de treino da camada.
	 */
	protected boolean treinando = false;

	/**
	 * Identificador único da camada.
	 */
	public int id;

	/**
	 * Instancia a camada base usada dentro dos modelos de Rede Neural.
	 * <p>
	 *    A camada base não possui implementação de métodos e é apenas usada
	 *    como molde de base para as outras camadas terem suas próprias implementações.
	 * </p>
	 */
	protected Camada() {}

	/**
	 * Monta a estrutura da camada.
	 * <p>
	 *		A construção da camada envolve inicializar seus atributos como 
	 *		entrada, kernels, bias, além de elementos auxiliares que são 
	 *		importantes para o seu funcionamento correto.
	 * </p>
	 * @param shape formato para os dados de entrada da camada.
	 */
	public abstract void construir(int[] shape);

	/**
	 * Verificador de inicialização para evitar problemas.
	 */
	protected void verificarConstrucao() {
		if (!_construida) {
			throw new IllegalStateException(
				"\nCamada " + nome() + " (id = " + id + ") não foi construída."
			);
		}
	}

	/**
	 * Inicaliza os parâmetros treináveis da camada de acordo com os 
	 * inicializadores definidos.
	 */
	public abstract void inicializar();

	/**
	 * Configura a função de ativação da camada através de uma instância 
	 * de {@code Ativacao} que será usada para ativar seus neurônios.
	 * <p>
	 *    Ativações disponíveis:
	 * </p>
	 * <ul>
	 *    <li> ReLU. </li>
	 *    <li> Sigmoid. </li>
	 *    <li> TanH. </li>
	 *    <li> Leaky ReLU. </li>
	 *    <li> ELU .</li>
	 *    <li> Swish. </li>
	 *    <li> GELU. </li>
	 *    <li> Linear. </li>
	 *    <li> Seno. </li>
	 *    <li> Argmax. </li>
	 *    <li> Softmax. </li>
	 *    <li> Softplus. </li>
	 *    <li> ArcTan. </li>
	 * </ul>
	 * <p>
	 *    Configurando a ativação da camada usando uma instância de função 
	 *    de ativação aumenta a liberdade de personalização dos hiperparâmetros
	 *    que algumas funções podem ter.
	 * </p>
	 * @param atv nova função de ativação.
	 */
	public void setAtivacao(Object atv) {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui configuração de função de ativação."
		);    
	}

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
	 * Configura uma seed fixa para geradores de números aleatórios da
	 * camada.
	 * @param seed nova seed.
	 */
	public void setSeed(Number seed) {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui configuração de seed."
		); 
	}

	/**
	 * Configura os nomes dos tensores usados pela camada, com intuito estético
	 * e de debug
	 */
	protected void setNomes() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui configuração para nomes de atributos."
		);  
	}

	/**
	 * Propaga os dados de entrada pela camada.
	 * @param entrada dados de entrada que serão processados pela camada.
	 * @return {@code Tensor} contendo a saída calculada pela camada.
	 */
	public abstract Tensor forward(Object entrada);

	/**
	 * Retropropaga os gradientes recebidos para as camadas anteriores.
	 * @param grad gradiente em relação a saída da camada.
	 * @return {@code Tensor} contendo os gradientes em relação a entrada da camada.
	 */
	public abstract Tensor backward(Object grad);

	/**
	 * Retorna a saída da camada.
	 * @return saída da camada.
	 */
	public abstract Tensor saida();

	/**
	 * Retorna a função de ativação configurada pela camada.
	 * @return função de ativação da camada.
	 */
	public Ativacao ativacao() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui função de ativação."
		);  
	}
	
	/**
	 * Retorna o formado dos dados de entrada suportados pela camada.
	 * @return formato de entrada da camada.
	 */
	public abstract int[] shapeEntrada();

	/**
	 * Retorna o formado dos dos dados de saída gerados pela camada.
	 * @return formato de saída da camada.
	 */
	public abstract int[] shapeSaida();

	/**
	 * Retorna a saída da camada no formato de array.
	 * @return saída da camada.
	 */
	public Variavel[] saidaParaArray() {
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
	 * Retorna o kernel da camada.
	 * <p>
	 *    <strong> O kernel só existe em camadas treináveis </strong>.
	 * </p>
	 * @return {@code Tensor} de kernel da camada.
	 */
	public Tensor kernel() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui kernel."
		);
	}

	/**
	 * Retorna um array contendo os elementos do kernel presentes na camada.
	 * <p>
	 *    <strong> O kernel só existe em camadas treináveis </strong>.
	 * </p>
	 * @return kernel da camada em formato de array.
	 */
	public Variavel[] kernelParaArray() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui kernel."
		);  
	}

	/**
	 * Retorna o gradiente do kernel da camada.
	 * <p>
	 *    <strong> O gradiente do kernel só existe em camadas treináveis </strong>.
	 * </p>
	 * @return {@code Tensor} de gradiente em relação ao kernel da camada.
	 */
	public Tensor gradKernel() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui gradiente de kernel."
		);
	}

	/**
	 * Retorna um array contendo os elementos usados para armazenar o valor
	 * dos gradientes para os kernels da camada.
	 * @return gradientes em relação aos kernels da camada em formato de array.
	 */
	public Variavel[] gradKernelParaArray() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui gradiente para kernel."
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
	 * Retorna um array contendo os elementos dos bias presente na camada.
	 * <p>
	 *    <strong> O bias só existe em camadas treináveis </strong>.
	 * </p>
	 * @return bias da camada em formato de array.
	 */
	public Variavel[] biasParaArray() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui bias."
		);    
	}

	/**
	 * Retorna os gradientes em relação ao bias da camada.
	 * @return {@code Tensor} de gradientes em relação ao bias da camada.
	 */
	public Tensor gradBias() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui gradiente para bias."
		);      
	}

	/**
	 * Retorna um array contendo os elementos usados para armazenar o valor
	 * dos gradientes para os bias da camada.
	 * @return gradientes em relação aos bias da camada em formato de array.
	 */
	public Variavel[] gradBiasParaArray() {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui gradiente para bias."
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
	 * Atribui os valores do kernel usando os valores contidos no array
	 * fornecido.
	 * @param kernel novos valores do kernel.
	 */
	public void setKernel(Variavel[] kernel) {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui edição de kernel."
		); 
	}

	/**
	 * Atribui os valores dos gradientes para o kernel usando os valores 
	 * contidos no array fornecido.
	 * @param grads novos valores de gradientes.
	 */
	public void setGradKernel(Variavel[] grads) {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui edição de gradiente para kernel."
		);    
	}

	/**
	 * Atribui os valores do bias usando os valores contidos no array
	 * fornecido.
	 * @param bias novos valores do bias.
	 */
	public void setBias(Variavel[] bias) {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui edição de bias."
		); 
	}

	/**
	 * Atribui os valores dos gradientes para o bias usando os valores 
	 * contidos no array fornecido.
	 * @param grads novos valores de gradientes.
	 */
	public void setGradBias(Variavel[] grads) {
		throw new UnsupportedOperationException(
			"\nCamada " + nome() + " não possui edição de gradiente para bias."
		);  
	}

	/**
	 * Zera os gradientes de parâmetros treináveis.
	 */
	public void zerarGrad() {
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
			"\nCamada " + nome() + " não possui exibição de informações."
		);
	}
}
