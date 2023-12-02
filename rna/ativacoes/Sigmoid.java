package rna.ativacoes;

import rna.estrutura.Convolucional;
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
            grad = camada.gradSaida.dado(i, j);
            d = camada.saida.dado(i, j);
            d = d * (1 - d);

            camada.derivada.editar(i, j, (grad * d));
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
               camada.saida[i].editar(j, k, sigmoid(s));
            }
         }
      }
   }

   private double sigmoid(double x){
      return 1 / (1 + Math.exp(-x));
   }
}
