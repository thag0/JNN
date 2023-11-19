package rna.avaliacao.perda;

public class EntropiaCruzada extends Perda{
   double eps = 1e-7;//evitar log 0

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
      
      double ec = 0.0;
      for(int i = 0; i < real.length; i++){
         ec += real[i] * Math.log(previsto[i] + eps);
      }
      
      return -ec;
   }
   
   @Override
   public double[] derivada(double[] previsto, double[] real){
      verificarDimensoes(previsto.length, real.length);

      //adaptação pra minha arquitetura por enquanto
      //não econtrei ainda uma boa resposta de como calcular isso
      double[] derivadas = new double[previsto.length];
      for(int i = 0; i < previsto.length; i++){
         derivadas[i] = real[i] - previsto[i];
      }
      return derivadas;
   }
}
