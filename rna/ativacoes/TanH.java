package rna.ativacoes;

import rna.estrutura.Convolucional;
import rna.estrutura.Densa;

public class TanH extends Ativacao{

   @Override
   public void calcular(Densa camada){
      double s;
      int i, j;

      for(i = 0; i < camada.saida.lin; i++){
         for(j = 0; j < camada.saida.col; j++){
            s = camada.somatorio.dado(i, j);
            camada.saida.editar(i, j, tanh(s));
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

   @Override
   public void calcular(Convolucional camada){
      int i, j, k;
      double s;

      for(i = 0; i < camada.somatorio.length; i++){
         for(j = 0; j < camada.somatorio[i].lin; j++){
            for(k = 0; k < camada.somatorio[i].col; k++){
               s = camada.somatorio[i].dado(j, k);
               camada.saida[i].editar(j, k, tanh(s));
            }
         }
      }
   }

   private double tanh(double x){
      return 2 / (1 + Math.exp(-2 * x)) - 1;
   }
}
