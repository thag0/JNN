package rna.avaliacao.perda;

public class ErroMedioQuadrado extends Perda{

   @Override
   public double calcular(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);
      
      int amostras = previsto.length;
      double emq = 0.0;
      for(int i = 0; i < amostras; i++){
         double d = real[i] - previsto[i];
         emq += d * d;
      }
      emq /= amostras;
      
      return emq;
   }
   
   @Override
   public double[] derivada(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);
      
      int amostras = previsto.length;
      double[] derivadas = new double[previsto.length];
      for(int i = 0; i < amostras; i++){
         derivadas[i] = 2 * (real[i] - previsto[i]) / amostras;
      }

      return derivadas;
   }
}
