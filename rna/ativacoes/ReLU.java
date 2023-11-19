package rna.ativacoes;

import rna.estrutura.CamadaDensa;

public class ReLU extends Ativacao{

   @Override
   public void calcular(CamadaDensa camada){
      for(int i = 0; i < camada.saida.length; i++){
         for(int j = 0; j < camada.saida[i].length; j++){
            camada.saida[i][j] = relu(camada.somatorio[i][j]);
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      for(int i = 0; i < camada.derivada.length; i++){
         for(int j = 0; j < camada.derivada[i].length; j++){
            camada.derivada[i][j] = derivada(camada.somatorio[i][j]);
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
