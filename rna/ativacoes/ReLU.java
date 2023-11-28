package rna.ativacoes;

import rna.estrutura.CamadaDensa;

public class ReLU extends Ativacao{

   @Override
   public void calcular(CamadaDensa camada){
      int i, j;
      double s;

      for(i = 0; i < camada.saida.lin; i++){
         for(j = 0; j < camada.saida.col; j++){
            s = relu(camada.somatorio.dado(i, j));
            camada.saida.editar(i, j, s);
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      int i, j;
      double grad, d;

      for(i = 0; i < camada.derivada.lin; i++){
         for(j = 0; j < camada.derivada.col; j++){
            grad = camada.gradienteSaida.dado(i, j);
            d = derivada(camada.somatorio.dado(i, j));
            
            camada.derivada.editar(i, j, (grad * d));
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
