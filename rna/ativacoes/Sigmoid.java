package rna.ativacoes;

import rna.estrutura.CamadaDensa;

public class Sigmoid extends Ativacao{

   @Override
   public void calcular(CamadaDensa camada){
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
   public void derivada(CamadaDensa camada){
      double sig;
      int i, j;

      for(i = 0; i < camada.saida.lin; i++){
         for(j = 0; j < camada.saida.col; j++){
            sig = camada.saida.dado(i, j);
            sig = sig * (1 - sig);
            camada.derivada.editar(i, j, sig);
         }
      }
   }

   private double sigmoid(double x){
      return 1 / (1 + Math.exp(-x));
   }
}
