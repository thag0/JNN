package rna.ativacoes;

import rna.estrutura.Convolucional;
import rna.estrutura.Densa;

public class ReLU extends Ativacao{

   @Override
   public void calcular(Densa camada){
      int i, j;
      double s;

      for(i = 0; i < camada.saida.lin; i++){
         for(j = 0; j < camada.saida.col; j++){
            s = relu(camada.somatorio.dado(i, j));
            camada.saida.editar(i, j, s);
         }
      }
   }

   @Override
   public void derivada(Densa camada){
      int i, j;
      double grad, d;

      for(i = 0; i < camada.derivada.lin; i++){
         for(j = 0; j < camada.derivada.col; j++){
            grad = camada.gradSaida.dado(i, j);
            d = camada.somatorio.dado(i, j);
            d = derivada(d);
            
            camada.derivada.editar(i, j, (grad * d));
         }
      }
   }

   @Override
   public void calcular(Convolucional camada){
      int i, j, k;
      double s;

      for(i = 0; i < camada.saida.length; i++){
         for(j = 0; j < camada.saida[i].lin; j++){
            for(k = 0; k < camada.saida[i].col; k++){
               s = camada.somatorio[i].dado(j, k);
               camada.saida[i].editar(j, k, relu(s));
            }
         }
      }
   }

   @Override
   public void derivada(Convolucional camada){
      int i, j, k;
      double grad, d;

      for(i = 0; i < camada.saida.length; i++){
         for(j = 0; j < camada.saida[i].lin; j++){
            for(k = 0; k < camada.saida[i].col; k++){
               grad = camada.gradSaida[i].dado(j, k);
               d = camada.somatorio[i].dado(j, k);
               d = derivada(d);

               camada.derivada[i].editar(j, k, (grad * d));
            }
         }
      }
   }

   private double relu(double x){
      return x > 0 ? x : 0;
   }

   private double derivada(double x){
      return x > 0 ? 1 : 0;
   }
}
