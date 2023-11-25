package rna.ativacoes;

import rna.estrutura.CamadaDensa;

public class TanH extends Ativacao{

   @Override
   public void calcular(CamadaDensa camada){
      double tanh;

      for(int i = 0; i < camada.saida.lin; i++){
         for(int j = 0; j < camada.saida.col; j++){
            tanh = camada.somatorio.dado(i, j);
            tanh = tanh(tanh);
            camada.saida.editar(i, j, tanh);
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      double tanh;

      for(int i = 0; i < camada.saida.lin; i++){
         for(int j = 0; j < camada.saida.col; j++){
            tanh = camada.saida.dado(i, j);
            tanh = 1 - (tanh * tanh);
            camada.derivada.editar(i, j, tanh);
         }
      }
   }

   private double tanh(double x){
      return 2 / (1 + Math.exp(-2*x)) -1;
   }
}
