package rna.ativacoes;

import rna.estrutura.Convolucional;
import rna.estrutura.Densa;

public class ReLU extends Ativacao{

   @Override
   public void calcular(Densa camada){
      super.aplicarFuncao(camada.somatorio, this::relu, camada.saida);
   }

   @Override
   public void derivada(Densa camada){
      super.aplicarDerivada(camada.gradSaida, camada.somatorio, this::relud, camada.derivada);
   }

   @Override
   public void calcular(Convolucional camada){
      int i, j, k;
      double s;

      for(i = 0; i < camada.saida.length; i++){
         for(j = 0; j < camada.saida[i].lin(); j++){
            for(k = 0; k < camada.saida[i].col(); k++){
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
         for(j = 0; j < camada.saida[i].lin(); j++){
            for(k = 0; k < camada.saida[i].col(); k++){
               grad = camada.gradSaida[i].dado(j, k);
               d = camada.somatorio[i].dado(j, k);
               d = relud(d);

               camada.derivada[i].editar(j, k, (grad * d));
            }
         }
      }
   }

   private double relu(double x){
      return x > 0 ? x : 0;
   }

   private double relud(double x){
      return x > 0 ? 1 : 0;
   }
}
