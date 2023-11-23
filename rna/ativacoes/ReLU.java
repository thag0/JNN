package rna.ativacoes;

import rna.estrutura.CamadaDensa;

public class ReLU extends Ativacao{

   @Override
   public void calcular(CamadaDensa camada){
      for(int i = 0; i < camada.saida.lin; i++){
         for(int j = 0; j < camada.saida.col; j++){
            camada.saida.editar(i, j, relu(camada.somatorio.dado(i, j)));
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      for(int i = 0; i < camada.derivada.lin; i++){
         for(int j = 0; j < camada.derivada.col; j++){
            camada.derivada.editar(i, j, derivada(camada.somatorio.dado(i, j)));
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
