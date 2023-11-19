package rna.ativacoes;

import rna.estrutura.CamadaDensa;

/**
 * Implementação da função de ativação GELU para uso dentro 
 * da {@code Rede Neural}.
 */
public class GELU extends Ativacao{

   /**
    * Intancia uma nova função de ativação GELU.
    */
   public GELU(){

   }

   @Override
   public void calcular(CamadaDensa camada){
      for(int i = 0; i < camada.saida.length; i++){
         for(int j = 0; j < camada.saida[i].length; j++){
            camada.saida[i][j] = gelu(camada.somatorio[i][j]);
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      for(int i = 0; i < camada.derivada.length; i++){
         for(int j = 0; j < camada.derivada[i].length; j++){
            camada.derivada[i][j] = derivada(camada.somatorio[i][j]);
         }
      }
   }

   private double gelu(double x){
      return x = 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));  
   }

   private double derivada(double x){
      double cdf;
      cdf = 0.5 * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
      return 0.5 * (1.0 + cdf + x * Math.exp(-Math.pow(x, 2) / 2.0) / Math.sqrt(2.0 * Math.PI));
   }
}
