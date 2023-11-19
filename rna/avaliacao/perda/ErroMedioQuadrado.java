package rna.avaliacao.perda;

public class ErroMedioQuadrado extends Perda{

   private void verificarDimensoes(int tamPrevisto, int tamReal){
      if(tamPrevisto != tamReal){
         throw new IllegalArgumentException(
            "Dimensões de dados previstos (" + tamPrevisto + 
            ") diferente da dimensão dos dados reais (" + tamReal + 
            ")"
         );
      }
   }

   @Override
   public double calcular(double[] previsto, double[] real){
      verificarDimensoes(previsto.length, real.length);
      
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
      verificarDimensoes(previsto.length, real.length);
      
      double[] derivadas = new double[previsto.length];
      for(int i = 0; i < previsto.length; i++){
         derivadas[i] = 2 * (real[i] - previsto[i]);
      }

      return derivadas;
   }
}
