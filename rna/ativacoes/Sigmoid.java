package rna.ativacoes;

import rna.estrutura.CamadaDensa;

public class Sigmoid extends Ativacao{

   @Override
   public void calcular(CamadaDensa camada){
      for(int i = 0; i < camada.saida.length; i++){
         for(int j = 0; j < camada.saida[i].length; j++){
            camada.saida[i][j] = sigmoid(camada.somatorio[i][j]);
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      for(int i = 0; i < camada.derivada.length; i++){
         for(int j = 0; j < camada.derivada[i].length; j++){
            double sig = camada.saida[i][j];
            camada.derivada[i][j] = sig * (1 - sig);
         }
      }
   }

   private double sigmoid(double x){
      return 1 / (1 + Math.exp(-x));
   }
}
