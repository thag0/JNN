package rna.estrutura;

import rna.ativacoes.Ativacao;
import rna.inicializadores.Inicializador;

/**
 * Classe base para as camadas dentro dos modelos de Rede Neural.
 * Novas camadas devem implementar os métodos padrões da classe Camada.
 */
public class Camada{

   /**
    * Controlador para uso dentro dos algoritmos de treino.
    */
   public boolean treinavel = false;

   /**
    * Controlador de construção da camada.
    */
   public boolean construida = false;

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
   public Camada(){

   }

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
   public void construir(Object entrada){
      throw new IllegalArgumentException(
         "Implementar construção da camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Inicaliza os parâmetros treináveis da camada, 
    * @param iniKernel inicializador para o kernel.
    * @param iniBias inicializador de bias.
    * @param x valor usado pelos inicializadores, dependendo do que for usado
    * pode servir de alcance na aleatorização, valor de constante, entre outros.
    */
   public void inicializar(Inicializador iniKernel, Inicializador iniBias, double x){
      throw new IllegalArgumentException(
         "Implementar inicialização da camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Inicaliza os pesos da camada de acordo com o inicializador configurado.
    * @param iniKernel inicializador para o kernel.
    * @param x valor usado pelos inicializadores, dependendo do que for usado
    * pode servir de alcance na aleatorização, valor de constante, entre outros.
    */
   public void inicializar(Inicializador iniKernel, double x){
      throw new IllegalArgumentException(
         "Implementar inicialização da camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Configura a função de ativação da camada através do nome fornecido, letras maiúsculas 
    * e minúsculas não serão diferenciadas.
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
    * @param ativacao nome da nova função de ativação.
    * @throws IllegalArgumentException se o valor fornecido não corresponder a nenhuma 
    * função de ativação suportada.
    */
   public void configurarAtivacao(String ativacao){
      throw new IllegalArgumentException(
         "Implementar configuração da função de ativação da camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Configura a função de ativação da camada através de uma instância de 
    * {@code FuncaoAtivacao} que será usada para ativar seus neurônios.
    * <p>
    *    Configurando a ativação da camada usando uma instância de função 
    *    de ativação aumenta a liberdade de personalização dos hiperparâmetros
    *    que algumas funções podem ter.
    * </p>
    * @param ativacao nova função de ativação.
    * @throws IllegalArgumentException se a função de ativação fornecida for nula.
    */
   public void configurarAtivacao(Ativacao ativacao){
      throw new IllegalArgumentException(
         "Implementar configuração da função de ativação da camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Configura o id da camada. O id deve indicar dentro da rede neural, em 
    * qual posição a camada está localizada.
    * @param id id da camada.
    */
   public void configurarId(int id){
      throw new IllegalArgumentException(
         "Implementar configuração de identificador da camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Configura o uso do bias para a camada.
    * <p>
    *    A configuração deve ser feita antes da construção da camada.
    * </p>
    * @param usarBias uso do bias.
    */
   public void configurarBias(boolean usarBias){
      throw new IllegalArgumentException(
         "Implementar configuração de bias da camada " + this.getClass().getTypeName() + "."
      );     
   }

   /**
    * Lógica para o processamento dos dados recebidos pela camada.
    * <p>
    *    Aqui as classes devem propagar os dados recebidos para
    *    as suas saídas.
    * </p>
    * O método deve levar em consideração o uso das funções de ativação
    * diretamente no seu processo de propagação.
    * @param entrada dados de entrada que poderão ser processados pela camada.
    */
   public void calcularSaida(Object entrada){
      throw new IllegalArgumentException(
         "Implementar cálculo de saída da camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Lógica para o cálculos dos gradientes de parâmetros treináveis dentro
    * da camada.
    * <p>
    *    Aqui as classes devem retropropagar os gradientes vindos da camada
    *    posterior, os usando para calcular seus próprios gradientes de parâmetros
    *    treinaveis (kernels, bias, etc).
    * </p>
    * O método deve levar em consideração o uso das funções de ativação
    * diretamente no seu processo de retropropagação.
    * @param gradSeguinte
    */
   public void calcularGradiente(Object gradSeguinte){
      throw new IllegalArgumentException(
         "Implementar cálculo de gradientes da camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Retorna a função de ativação configurada pela camada.
    * @return função de ativação da camada.
    */
   public Ativacao obterAtivacao(){
      throw new IllegalArgumentException(
         "Implementar retorno da função de ativação da camada " + this.getClass().getTypeName() + "."
      );
   }
   
   /**
    * Lógica para retornar o formato configurado de entrada da camada.
    * <p>
    *    Nele devem ser consideradas as dimensões dos dados de entrada da
    *    camada, que devem estar disposto como:
    * </p>
    * <pre>
    *    formato = (altura, largura, profundidade ...)
    * </pre>
    * @return array contendo os valores das dimensões de entrada da camada.
    */
   public int[] formatoEntrada(){
      throw new IllegalArgumentException(
         "Implementar formato de entrada da camada" + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Lógica para retornar o formato configurado de saída da camada.
    * <p>
    *    Nele devem ser consideradas as dimensões dos dados de saída da
    *    camada, que devem estar disposto como:
    * </p>
    * <pre>
    *    formato = (altura, largura, profundidade ...)
    * </pre>
    * @return array contendo os valores das dimensões de saída da camada.
    */
   public int[] formatoSaida(){
      throw new IllegalArgumentException(
         "Implementar formato de saída da camada" + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Retorna a saída da camada no formato de array.
    * @return saída da camada.
    */
   public double[] saidaParaArray(){
      throw new IllegalArgumentException(
         "Implementar retorno de saída para array da camada" + this.getClass().getTypeName() + "."
      );   
   }

   /**
    * Retorna a quantidade total de elementos presentes na saída da camada.
    * @return tamanho de saída da camada.
    */
   public int tamanhoSaida(){
      throw new IllegalArgumentException(
         "Implementar retorno de tamanho da saída da camada " + this.getClass().getTypeName() + "."
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
   public int numParametros(){
      throw new IllegalArgumentException(
         "Implementar número de parâmetros da camada " + this.getClass().getTypeName() + "."
      );  
   }

   /**
    * Retorna o verificador de uso do bias dentro da camada.
    * @return uso de bias na camada.
    */
   public boolean temBias(){
      throw new IllegalArgumentException(
         "Implementar uso do bias na camada " + this.getClass().getTypeName() + "."
      );  
   }

   /**
    * Retorna um array contendo os elementos do kernel presente na camada.
    * <p>
    *    O kernel de uma camada inclui seus atributos mais importantes, como
    *    os pesos de uma camada densa, ou os filtros de uma camada convolucional.
    * </p>
    * @return kernel da camada.
    */
   public double[] obterKernel(){
      throw new IllegalArgumentException(
         "Implementar retorno do kernel da camada " + this.getClass().getTypeName() + "."
      );       
   }

   /**
    * Retorna um array contendo os elementos usados para armazenar o valor
    * dos gradientes para os kernels da camada.
    * @return gradientes para os kernels da camada.
    */
   public double[] obterGradKernel(){
      throw new IllegalArgumentException(
         "Implementar retorno do gradiente para o kernel da camada" + this.getClass().getTypeName() + "."
      );       
   }

   /**
    * Retorna um array contendo os elementos dos bias presente na camada.
    * <p>
    *    É importante verificar se a camada foi configurada para suportar
    *    os bias antes de usar os valores retornados por ela. Quando não
    *    configurados, os bias da camada são nulos.
    * </p>
    * @return bias da camada.
    */
   public double[] obterBias(){
      throw new IllegalArgumentException(
         "Implementar retorno do bias da camada " + this.getClass().getTypeName() + "."
      );        
   }

   /**
    * Retorna um array contendo os elementos usados para armazenar o valor
    * dos gradientes para os bias da camada.
    * @return gradientes para os bias da camada.
    */
   public double[] obterGradBias(){
      throw new IllegalArgumentException(
         "Implementar retorno do gradiente para o bias da camada" + this.getClass().getTypeName() + "."
      );        
   }

   /**
    * Retorna o gradiente de entrada da camada, dependendo do tipo
    * de camada, esse gradiente pode assumir diferentes tipos de objetos.
    * @return gradiente de entrada da camada.
    */
   public Object obterGradEntrada(){
      throw new IllegalArgumentException(
         "Implementar retorno do gradiente de entrada da camada " + this.getClass().getTypeName() + "."
      );     
   }

   /**
    * Ajusta os valores do kernel usando os valores contidos no array
    * fornecido.
    * @param kernel novos valores do kernel.
    */
   public void editarKernel(double[] kernel){
      throw new IllegalArgumentException(
         "Implementar edição do kernel para a camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Ajusta os valores do bias usando os valores contidos no array
    * fornecido.
    * @param bias novos valores do bias.
    */
   public void editarBias(double[] bias){
      throw new IllegalArgumentException(
         "Implementar edição do bias para a camada " + this.getClass().getTypeName() + "."
      );
   }

   /**
    * Clona as características principais da camada.
    * @return clone da camada.
    */
   public Camada clonar(){
      throw new IllegalArgumentException(
         "Implementar clonagem para a camada " + this.getClass().getTypeName() + "." 
      );
   }
}
