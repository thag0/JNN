package rna.camadas;

import rna.ativacoes.Ativacao;
import rna.core.Tensor4D;

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
public abstract class Camada{

   /**
    * Controlador para uso dentro dos algoritmos de treino.
    */
   protected boolean treinavel = false;

   /**
    * Controlador de construção da camada.
    */
   public boolean construida = false;

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
   protected Camada(){}

   /**
    * Monta a estrutura da camada.
    * <p>
    *    A construção da camada envolve inicializar seus atributos como entrada,
    *    kernels, bias, além de elementos auxiliares que são importantes para
    *    o seu funcionamento correto.
    * </p>
    * @param entrada formato de entrada da camada, dependerá do formato de saída
    * da camada anterior, no caso de ser a primeira camada, dependerá do formato
    * dos dados de entrada.
    */
   public abstract void construir(Object entrada);

   /**
    * Verificador de inicialização para evitar problemas.
    */
   protected void verificarConstrucao(){
      if(this.construida == false){
         throw new IllegalArgumentException(
            "\nCamada " + nome() + " (id = " + this.id + ") não foi construída."
         );
      }
   }

   /**
    * Inicaliza os parâmetros treináveis da camada de acordo com os inicializadores
    * definidos.
    */
   public abstract void inicializar();

   /**
    * Configura a função de ativação da camada através de uma instância de 
    * {@code Ativacao} que será usada para ativar seus neurônios.
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
   public void setAtivacao(Object atv){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui configuração de função de ativação."
      );    
   }

   /**
    * Configura o id da camada. O id deve indicar dentro de um modelo, em 
    * qual posição a camada está localizada.
    * @param id id da camada.
    */
   public void setId(int id){
      if(id < 0){
         throw new IllegalArgumentException(
            "\nId da camada deve ser maior ou igual a zero, recebido: " + id + "."
         );
      }

      this.id = id;
   }

   /**
    * Configura o uso do bias para a camada.
    * <p>
    *    A configuração deve ser feita antes da construção da camada.
    * </p>
    * @param usarBias uso do bias.
    */
   public void setBias(boolean usarBias){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui configuração de bias."
      );    
   }

   /**
    * Configura a camada para treino.
    * @param treinando caso verdadeiro a camada será configurada para
    * treino, caso contrário, será usada para testes/predições.
    */
   public void setTreino(boolean treinando){
      this.treinando = true;
   }

   /**
    * Configura uma seed fixa para geradores de números aleatórios da
    * camada.
    * @param seed nova seed.
    */
   public void setSeed(long seed){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui configuração de seed."
      ); 
   }

   /**
    * Configura os nomes dos tensores usados pela camada, com intuito estético
    * e de debug
    */
   protected void setNomes(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui configuração para nomes de atributos."
      );  
   }

   /**
    * Propaga os dados de entrada pela camada.
    * @param entrada dados de entrada que serão processados pela camada.
    * @return {@code Tensor} contendo a saída calculada pela camada.
    */
   public abstract Tensor4D forward(Object entrada);

   /**
    * Retropropaga os gradientes recebidos para as camadas anteriores.
    * @param grad gradiente em relação a saída da camada.
    * @return {@code Tensor} contendo os gradientes em relação a entrada da camada.
    */
   public abstract Tensor4D backward(Object grad);

   /**
    * Retorna a saída da camada.
    * @return saída da camada.
    */
   public abstract Tensor4D saida();

   /**
    * Retorna a função de ativação configurada pela camada.
    * @return função de ativação da camada.
    */
   public Ativacao ativacao(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui função de ativação."
      );  
   }
   
   /**
    * Lógica para retornar o formato configurado de entrada da camada.
    * <p>
    *    Nele devem ser consideradas as dimensões dos dados de entrada da
    *    camada, que devem estar disposto como:
    * </p>
    * <pre>
    *    formato = (profundidade, altura, largura)
    * </pre>
    * @return array contendo os valores das dimensões de entrada da camada.
    */
   public abstract int[] formatoEntrada();

   /**
    * Lógica para retornar o formato configurado de saída da camada.
    * <p>
    *    Nele devem ser consideradas as dimensões dos dados de saída da
    *    camada, que devem estar disposto como:
    * </p>
    * <pre>
    *    formato = (profundidade, altura, largura)
    * </pre>
    * @return array contendo os valores das dimensões de saída da camada.
    */
   public abstract int[] formatoSaida();

   /**
    * Retorna a saída da camada no formato de array.
    * @return saída da camada.
    */
   public double[] saidaParaArray(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui retorno de saída para array."
      );    
   }

   /**
    * Retorna a quantidade total de elementos presentes na saída da camada.
    * @return tamanho de saída da camada.
    */
   public int tamanhoSaida(){
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
   public abstract int numParametros();

   /**
    * Retorna o verificador de uso do bias dentro da camada.
    * @return uso de bias na camada.
    */
   public boolean temBias(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui verificação de bias."
      );
   }

   /**
    * Retorna o kernel da camada.
    * <p>
    *    O kernel de uma camada inclui seus atributos mais importantes, como
    *    os pesos de uma camada densa, ou os filtros de uma camada convolucional.
    * </p>
    * <p>
    *    <strong> O kernel só existe em camadas treináveis </strong>.
    * </p>
    * @return kernel da camada.
    */
   public Tensor4D kernel(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui kernel."
      );
   }

   /**
    * Retorna um array contendo os elementos do kernel presente na camada.
    * <p>
    *    O kernel de uma camada inclui seus atributos mais importantes, como
    *    os pesos de uma camada densa, ou os filtros de uma camada convolucional.
    * </p>
    * <p>
    *    <strong> O kernel só existe em camadas treináveis </strong>.
    * </p>
    * @return kernel da camada.
    */
   public double[] kernelParaArray(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui kernel."
      );  
   }

   /**
    * Retorna o gradiente do kernel da camada.
    * <p>
    *    <strong> O gradiente do kernel só existe em camadas treináveis </strong>.
    * </p>
    * @return gradiente do kernel da camada.
    */
   public Tensor4D gradKernel(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui gradiente de kernel."
      );
   }

   /**
    * Retorna um array contendo os elementos usados para armazenar o valor
    * dos gradientes para os kernels da camada.
    * @return gradientes para os kernels da camada.
    */
   public double[] gradKernelParaArray(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui gradiente para kernel."
      );   
   }

   /**
    * Retorna o bias da camada.
    * <p>
    *    É importante verificar se a camada foi configurada para suportar
    *    os bias antes de usar os valores retornados por ela. Quando não
    *    configurados, os bias da camada são nulos.
    * </p>
    * <p>
    *    <strong> O bias só existe em camadas treináveis </strong>.
    * </p>
    * @return bias da camada.
    */
   public Tensor4D bias(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui bias."
      );      
   }

   /**
    * Retorna um array contendo os elementos dos bias presente na camada.
    * <p>
    *    É importante verificar se a camada foi configurada para suportar
    *    os bias antes de usar os valores retornados por ela. Quando não
    *    configurados, os bias da camada são nulos.
    * </p>
    * <p>
    *    <strong> O bias só existe em camadas treináveis </strong>.
    * </p>
    * @return bias da camada.
    */
   public double[] biasParaArray(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui bias."
      );    
   }

   /**
    * Retorna um array contendo os elementos usados para armazenar o valor
    * dos gradientes para os bias da camada.
    * @return gradientes para os bias da camada.
    */
   public double[] gradBias(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui gradiente para bias."
      );      
   }

   /**
    * Retorna o gradiente de entrada da camada, dependendo do tipo
    * de camada, esse gradiente pode assumir diferentes tipos de objetos.
    * @return {@code Tensor} contendo o gradiente de entrada da camada.
    */
   public Tensor4D gradEntrada(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui gradiente de entrada."
      );    
   }

   /**
    * Ajusta os valores do kernel usando os valores contidos no array
    * fornecido.
    * @param kernel novos valores do kernel.
    */
   public void setKernel(double[] kernel){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui edição de kernel."
      ); 
   }

   /**
    * Ajusta os valores dos gradientes para o kernel usando os valores 
    * contidos no array fornecido.
    * @param grads novos valores de gradientes.
    */
   public void setGradienteKernel(double[] grads){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui edição de gradiente para kernel."
      );    
   }

   /**
    * Ajusta os valores do bias usando os valores contidos no array
    * fornecido.
    * @param bias novos valores do bias.
    */
   public void setBias(double[] bias){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui edição de bias."
      ); 
   }

   /**
    * Ajusta os valores dos gradientes para o bias usando os valores 
    * contidos no array fornecido.
    * @param grads novos valores de gradientes.
    */
   public void setGradienteBias(double[] grads){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui edição de gradiente para bias."
      );  
   }

   /**
    * Zera os gradientes para os kernels e bias da camada.
    */
   public void zerarGradientes(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui kernel/bias para zerar."
      );
   }

   /**
    * Verifica se a camada é treinável.
    * @return {@code true} caso a camada seja treinável, {@code false},
    * caso contrário.
    */
   public boolean treinavel(){
      return treinavel;
   }

   /**
    * Clona as características principais da camada.
    * @return clone da camada.
    */
   public Camada clonar(){
      throw new UnsupportedOperationException(
         "\nCamada " + nome() + " não possui suporte para clonagem."
      );
   }

   /**
    * Retorna o nome da camada.
    * @return nome da camada.
    */
   public String nome(){
      return getClass().getSimpleName();
   }
}
