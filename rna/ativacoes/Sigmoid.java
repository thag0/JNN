package rna.ativacoes;

import rna.estrutura.CamadaDensa;

public class Sigmoid extends Ativacao{

   @Override
   public void calcular(CamadaDensa camada){
      for(int i = 0; i < camada.saida.lin; i++){
         for(int j = 0; j < camada.saida.col; j++){
            camada.saida.editar(i, j, sigmoid(camada.somatorio.dado(i, j)));
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      for(int i = 0; i < camada.saida.lin; i++){
         for(int j = 0; j < camada.saida.col; j++){
            double sig = camada.saida.dado(i, j);
            camada.derivada.editar(i, j, (sig * (1 - sig)));
         }
      }
   }

   private double sigmoid(double x){
      return 1 / (1 + Math.exp(-x));
   }
}
