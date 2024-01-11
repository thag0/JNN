package rna.avaliacao.perda;

/**
 * Função de perda Mean Absolute Error, calcula o erro médio 
 * absoluto entre as previsões e os valores reais.
 */
public class MAE extends Perda{

   /**
    * Inicializa a função de perda Mean Absolute Error.
    */
   public MAE(){}

   @Override
   public double calcular(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);
      
      int amostras = previsto.length;
      double mae = 0;
      for(int i = 0; i < amostras; i++){
         mae += Math.abs(real[i] - previsto[i]);
      }
      mae /= amostras;
      
      return mae;
   }
   
   @Override
   public double[] derivada(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);

      double[] derivadas = new double[previsto.length];
      for(int i = 0; i < previsto.length; i++){
         derivadas[i] = real[i] - previsto[i];
      }
      return derivadas;
   }
}
