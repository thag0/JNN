package rna.ativacoes;

import rna.estrutura.Densa;

public class TanH extends Ativacao{

   @Override
   public void calcular(Densa camada){
      double s;
      int i, j;

      for(i = 0; i < camada.saida.lin; i++){
         for(j = 0; j < camada.saida.col; j++){
            s = camada.somatorio.dado(i, j);
            s = tanh(s);
            camada.saida.editar(i, j, s);
         }
      }
   }

   @Override
   public void derivada(Densa camada){
      double grad, d;
      int i, j;

      for(i = 0; i < camada.saida.lin; i++){
         for(j = 0; j < camada.saida.col; j++){
            grad = camada.gradienteSaida.dado(i, j);
            d = camada.saida.dado(i, j);
            d = 1 - (d * d);
            
            camada.derivada.editar(i, j, (d * grad));
         }
      }
   }

   private double tanh(double x){
      return 2 / (1 + Math.exp(-2 * x)) - 1;
   }
}
