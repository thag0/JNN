package rna.avaliacao.perda;

/**
 * Função de perda Mean Squared Error, calcula o erro médio 
 * quadrado entre as previsões e os valores reais.
 */
public class MSE extends Perda{

   /**
    * Inicializa a função de perda Mean Squared Error.
    */
   public MSE(){}

   @Override
   public double calcular(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);
      
      int amostras = previsto.length;
      double mse = 0.0;
      for(int i = 0; i < amostras; i++){
         double d = previsto[i] - real[i];
         mse += d * d;
      }
      mse /= amostras;
      
      return mse;
   }
   
   @Override
   public double[] derivada(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);
      
      int amostras = previsto.length;
      double[] derivadas = new double[previsto.length];
      for(int i = 0; i < amostras; i++){
         derivadas[i] = (2 / amostras) * (previsto[i] - real[i]);
      }

      return derivadas;
   }
}
