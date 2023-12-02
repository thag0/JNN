package rna.ativacoes;

import rna.estrutura.Densa;

public class ArcTan extends Ativacao{

   @Override
   public void calcular(Densa camada){
      int i, j;
      double s;

      for(i = 0; i < camada.saida.lin; i++) {
         for(j = 0; j < camada.saida.col; j++){
            s = camada.somatorio.dado(i, j);
            s = arctan(s);
            camada.saida.editar(i, j, s);
         }
      }
   }

   @Override
   public void derivada(Densa camada){
      int i, j;
      double d, grad;

      for(i = 0; i < camada.derivada.lin; i++){
         for(j = 0; j < camada.derivada.col; j++){
            d = camada.saida.dado(i, j);
            d = derivada(d);
            grad = camada.gradSaida.dado(i, j);
            camada.derivada.editar(i, j, (grad * d));
         }
      }
   }

   private double arctan(double x){
      return Math.atan(x);
   }

   private double derivada(double x){
      return 1.0 / (1.0 + (x * x));
   }
}
