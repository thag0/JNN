package rna.ativacoes;

import rna.estrutura.CamadaDensa;

/**
 * Implementação da função de ativação Linear para uso dentro 
 * da {@code Rede Neural}.
 */
public class Linear extends Ativacao{

   /**
    * Instancia a função de ativação Linear.
    */
   public Linear(){

   }

   @Override
   public void calcular(CamadaDensa camada){
      for(int i = 0; i < camada.saida.length; i++){
         for(int j = 0; j < camada.saida[i].length; j++){
            camada.saida[i][j] = camada.somatorio[i][j];
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      for(int i = 0; i < camada.derivada.length; i++){
         for(int j = 0; j < camada.derivada[i].length; j++){
            camada.derivada[i][j] = 1;
         }
      }
   }
}
