package rna.avaliacao.perda;

public class ErroMedioAbsoluto extends Perda{

   @Override
   public double calcular(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);
      
      int amostras = previsto.length;
      double ema = 0;
      for(int i = 0; i < amostras; i++){
         ema += Math.abs(real[i] - previsto[i]);
      }
      ema /= amostras;
      
      return ema;
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
