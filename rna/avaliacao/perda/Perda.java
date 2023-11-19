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
}
