package rna.ativacoes;

import rna.estrutura.Densa;

public class Sigmoid extends Ativacao{

   @Override
   public void calcular(Densa camada){
      double sig;
      int i, j;

      for(i = 0; i < camada.saida.lin; i++){
         for(j = 0; j < camada.saida.col; j++){
            sig = camada.somatorio.dado(i, j);
            sig = sigmoid(sig);
            camada.saida.editar(i, j, sig);
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
            d = d * (1 - d);

            camada.derivada.editar(i, j, (grad * d));
         }
      }
   }

   private double sigmoid(double x){
      return 1 / (1 + Math.exp(-x));
   }
}
