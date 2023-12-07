package rna.avaliacao.perda;

public class EntropiaCruzadaBinaria extends Perda{
   double eps = 1e-7;

   @Override
   public double calcular(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);

      double ecb = 0;
      for(int i = 0; i < real.length; i++){
         ecb += (real[i] * Math.log(previsto[i] + eps)) + (1 - real[i]) * (Math.log(1 - previsto[i] + eps));
      }

      return -ecb / real.length;
   }

   @Override
   public double[] derivada(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);
      double[] derivadas = new double[previsto.length];
      int n = derivadas.length;

      for(int i = 0; i < n; i++){
         derivadas[i] = (((1.0 - real[i]) / (1.0 - previsto[i])) - (real[i] / previsto[i])) / n;
      }
      return derivadas;
   }
}
