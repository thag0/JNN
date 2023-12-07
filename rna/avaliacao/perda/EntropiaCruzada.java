package rna.avaliacao.perda;

public class EntropiaCruzada extends Perda{
   double eps = 1e-15;//evitar log 0

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

      //tentando encontrar uma adaptação que funcione
      for(int i = 0; i < previsto.length; i++){
         derivadas[i] = real[i] - previsto[i];
      }
      return derivadas;
   }
}
