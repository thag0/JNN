package rna.avaliacao.perda;

/**
 * <h2>
 *    Base para implementações de funções de perda
 * </h2>
 * <p>
 *    As funções de perda são usadas para avaliações de modelos
 *    e principalmente para o treinamento, calculando os gradientes
 *    necessários para a atualização de parâmetros dos modelos.
 * </p>
 */
public abstract class Perda{

   /**
    * Calcula a função de perda configurada.
    * @param previsto dados previstos.
    * @param real dados rotulados.
    * @return valor de perda de acordo com a função configurada.
    */
   public abstract double calcular(double[] previsto, double[] real);

   /**
    * Calcula a derivada da função de perda configurada.
    * @param previsto dados previstos.
    * @param real dados rotulados.
    * @return valor de derivada de acordo com a função configurada.
    */
   public abstract double[] derivada(double[] previsto, double[] real);

   /**
    * Auxiliar para verificar se os tamanhos dos arrays que serão usados
    * pelas funções de perda são iguais.
    * @param previsto dados previstos.
    * @param real dados rotulados.
    */
   protected void verificarDimensoes(double[] previsto, double[] real){
      if(previsto.length != real.length){
         throw new IllegalArgumentException(
            "Dimensões de dados previstos (" + previsto.length + 
            ") diferente da dimensão dos dados reais (" + real.length + 
            ")"
         );
      }
   }

   /**
    * Retorna o nome da função de perda.
    * @return nome da função de perda.
    */
   public String nome(){
      return getClass().getSimpleName();
   }
}
