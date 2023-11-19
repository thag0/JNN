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
      for(int i = 0; i < camada.saida.length; i++){
         for(int j = 0; j < camada.saida[i].length; j++){
            camada.saida[i][j] = camada.somatorio[i][j] * sigmoid(camada.somatorio[i][j]);
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      for(int i = 0; i < camada.derivada.length; i++){
         for(int j = 0; j < camada.derivada[i].length; j++){
            double sig = sigmoid(camada.somatorio[i][j]);
            camada.derivada[i][j] = sig + (camada.somatorio[i][j] * sig * (1 - sig));
         }
      }
   }
}
