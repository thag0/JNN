package rna.ativacoes;

import rna.estrutura.CamadaDensa;

public class TanH extends Ativacao{

   @Override
   public void calcular(CamadaDensa camada){
      for(int i = 0; i < camada.saida.lin; i++){
         for(int j = 0; j < camada.saida.col; j++){
            camada.saida.editar(i, j, tanh(camada.somatorio.dado(i, j)));
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      for(int i = 0; i < camada.saida.lin; i++){
         for(int j = 0; j < camada.saida.col; j++){
            double tanh = camada.saida.dado(i, j);
            camada.derivada.editar(i, j, (1 - (tanh * tanh)));
         }
      }
   }

   private double tanh(double x){
      return 2 / (1 + Math.exp(-2*x)) -1;
   }
}
