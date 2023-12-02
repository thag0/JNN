package rna.estrutura;

import rna.ativacoes.Ativacao;
import rna.inicializadores.Inicializador;

/**
 * Classe base para as camadas dentro dos modelos de Rede Neural.
 * Novas camadas devem implementar os métodos padrões da classe Camada.
 */
public class Camada{

   public Camada(){

   }

   public void configurarId(int id){

   }

   public void inicializar(Inicializador iniKernel, Inicializador iniBias, double x){
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
}
