package rna.estrutura;

import rna.ativacoes.Ativacao;
import rna.inicializadores.Inicializador;

/**
 * Classe base para as camadas dentro dos modelos de Rede Neural.
 * Novas camadas devem implementar os métodos padrões da classe Camada.
 */
public class Camada{

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
    * Inicaliza os parâmetros treináveis da camada, 
    * @param iniKernel inicializador para o kernel.
    * @param iniBias inicializador de bias.
    * @param x valor usado pelos inicializadores, dependendo do que for usado
    * pode servir de alcance na aleatorização, valor de constante, entre outros.
    */
   public void inicializar(Inicializador iniKernel, Inicializador iniBias, double x){
      throw new IllegalArgumentException(
         "Implementar inicialização da camada."
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
         "Implementar inicialização da camada."
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
         "Implementar configuração da função de ativação da camada."
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
         "Implementar configuração da função de ativação da camada."
      );
   }

   /**
    * Configura o id da camada. O id deve indicar dentro da rede neural, em 
    * qual posição a camada está localizada.
    * @param id id da camada.
    */
   public void configurarId(int id){
      throw new IllegalArgumentException(
         "Implementar configuração de identificador."
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
         "Implementar cálculo de saída."
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
         "Implementar cálculo de gradientes."
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
         "Implementar formato de entrada."
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
         "Implementar formato de saída."
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
         "Implementar númedo de parâmetros."
      );  
   }

   /**
    * Retorna o verificador de uso do bias dentro da camada.
    * @return uso de bias na camada.
    */
   public boolean temBias(){
      throw new IllegalArgumentException(
         "Implementar uso do bias."
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
         "Implementar retorno do kernel."
      );       
   }

   public double[] obterGradKernel(){
      throw new IllegalArgumentException(
         "Implementar retorno do gradiente para o kernel."
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
         "Implementar retorno do bias."
      );        
   }

   public double[] obterGradBias(){
      throw new IllegalArgumentException(
         "Implementar retorno do gradiente para o bias."
      );        
   }
}
