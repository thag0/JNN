package rna.ativacoes;

import rna.estrutura.CamadaDensa;

public class TanH extends Ativacao{
   @Override
   public void calcular(CamadaDensa camada){
      for(int i = 0; i < camada.saida.length; i++){
         for(int j = 0; j < camada.saida[i].length; j++){
            camada.saida[i][j] = tanh(camada.somatorio[i][j]);
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      for(int i = 0; i < camada.derivada.length; i++){
         for(int j = 0; j < camada.derivada[i].length; j++){
            double tanh = camada.saida[i][j];
            camada.derivada[i][j] = 1 - (tanh * tanh);
         }
      }
   }

   private double tanh(double x){
      return 2 / (1 + Math.exp(-2*x)) -1;
   }
}
