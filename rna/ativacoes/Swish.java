package rna.ativacoes;

import rna.estrutura.CamadaDensa;

/**
 * Implementação da função de ativação Swish para uso 
 * dentro da {@code Rede Neural}.
 */
public class Swish extends Ativacao{

   /**
    * Instancia a função de ativação Swish.
    */
   public Swish(){

   }

   private double sigmoid(double x){
      return (1 / (1 + Math.exp(-x)) );
   }

   @Override
   public void calcular(CamadaDensa camada){
      for(int i = 0; i < camada.saida.lin; i++){
         for(int j = 0; j < camada.saida.col; j++){
            double s = camada.somatorio.dado(i, j) * sigmoid(camada.somatorio.dado(i, j));
            camada.saida.editar(i, j, s);
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      for(int i = 0; i < camada.derivada.lin; i++){
         for(int j = 0; j < camada.derivada.col; j++){
            double sig = sigmoid(camada.somatorio.dado(i, j));
            double d = sig + (camada.somatorio.dado(i, j) * sig * (1 - sig));
            camada.derivada.editar(i, j, d);
         }
      }
   }
}
