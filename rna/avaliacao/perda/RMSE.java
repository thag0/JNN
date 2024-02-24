package rna.avaliacao.perda;

/**
 * Função de perda Root Mean Squared Error, calcula a raiz do 
 * erro médio quadrado entre as previsões e os valores reais.
 */
public class RMSE extends Perda{
 
   /**
    * Inicializa a função de perda Root Mean Squared Error.
    */
   public RMSE(){}

   @Override
   public double calcular(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);
      
      int amostras = previsto.length;
      double rmse = 0.0;
      for(int i = 0; i < amostras; i++){
         double d = previsto[i] - real[i];
         rmse += d * d;
      }
      rmse /= amostras;
      
      return Math.sqrt(rmse);
   }
    
   @Override
   public double[] derivada(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);
      
      int amostras = previsto.length;
      double[] derivadas = new double[previsto.length];
      double rrmse = Math.sqrt(calcular(previsto, real));

      for(int i = 0; i < amostras; i++){
         derivadas[i] = (previsto[i] - real[i]) / (rrmse * amostras);
      }

      return derivadas;
   }
}
