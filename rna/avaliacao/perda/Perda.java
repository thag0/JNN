package rna.avaliacao.perda;

public abstract class Perda{

   /**
    * Calcula a função de perda configurada.
    * @param previsto dados previstos.
    * @param real dados rotulados.
    * @return valor de perda de acordo com a função configurada.
    */
   public double calcular(double[] previsto, double[] real){
      throw new UnsupportedOperationException(
         "É necessário implementar o cálculo de perda."
      );
   }

   /**
    * Calcula a derivada da função de perda configurada.
    * @param previsto dados previstos.
    * @param real dados rotulados.
    * @return valor de derivada de acordo com a função configurada.
    */
   public double[] derivada(double[] previsto, double[] real){
      throw new UnsupportedOperationException(
         "É necessário implementar o cálculo de derivada."
      );
   }

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
