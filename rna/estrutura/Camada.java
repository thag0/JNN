package rna.estrutura;

/**
 * Classe base para as camadas dentro dos modelos de Rede Neural.
 * Novas camadas devem implementar os métodos padrões da classe Camada.
 */
public class Camada{

   public Camada(){

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
