package rna.avaliacao.perda;

public class ErroMedioQuadradoLogaritmico extends Perda{

   @Override
   public double calcular(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);
      
      int amostras = previsto.length;
      double emql = 0;
      for(int i = 0; i < amostras; i++){
         double d = Math.log(1 + real[i]) - Math.log(1 + previsto[i]);
         emql += d * d;
      }
      emql /= amostras;
      
      return emql;
   }
   
   @Override
   public double[] derivada(double[] previsto, double[] real){
      super.verificarDimensoes(previsto, real);

      int amostras = previsto.length;
      double[] derivadas = new double[previsto.length];
      for(int i = 0; i < amostras; i++){
         derivadas[i] = 2 * (Math.log(1 + real[i]) - Math.log(1 + previsto[i])) / amostras;
      }
      return derivadas;
   }
}
