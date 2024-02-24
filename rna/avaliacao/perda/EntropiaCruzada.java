package rna.avaliacao.perda;

/**
 * Função de perda Cross Entropy, que é normalmente usada em problemas 
 * de classificação multiclasse. Ela mede a discrepância entre a distribuição 
 * de probabilidade prevista e a distribuição de probabilidade real dos rótulos.
 */
public class EntropiaCruzada extends Perda{
   double eps = 1e-7;//evitar log 0

   /**
    * Inicializa a função de perda Categorical Cross Entropy.
    */
   public EntropiaCruzada(){}

   @Override
   public double calcular(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);
      
      double ec = 0.0;
      for(int i = 0; i < real.length; i++){
         ec += real[i] * Math.log(previsto[i] + eps);
      }
      
      return -ec;
   }
   
   @Override
   public double[] derivada(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);
      double[] derivadas = new double[previsto.length];

      for(int i = 0; i < previsto.length; i++){
         derivadas[i] = previsto[i] - real[i];
      }

      return derivadas;
   }
}
