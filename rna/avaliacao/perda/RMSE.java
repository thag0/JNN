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
      double mse = 0.0;
      for(int i = 0; i < amostras; i++){
         double d = real[i] - previsto[i];
         mse += d * d;
      }
      mse /= amostras;
      
      return Math.sqrt(mse);
   }
    
   @Override
   public double[] derivada(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);
      
      int amostras = previsto.length;
      double rmse = this.calcular(previsto, real);
      double[] derivadas = new double[previsto.length];
      for(int i = 0; i < amostras; i++){
         derivadas[i] = (real[i] - previsto[i]) / (rmse * amostras);
      }

      return derivadas;
   }
}
