package rna.ativacoes;

import rna.estrutura.Convolucional;
import rna.estrutura.Densa;

public class ReLU extends Ativacao{

   public ReLU(){
      super.construir(this::relu, this::relud);
   }

   @Override
   public void calcular(Densa camada){
      super.aplicarFx(camada.somatorio, camada.saida);
   }

   @Override
   public void derivada(Densa camada){
      super.aplicarDx(camada.gradSaida, camada.somatorio, camada.derivada);
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
