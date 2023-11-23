package rna.ativacoes;

import rna.estrutura.CamadaDensa;

/**
 * Implementação da função de ativação SoftPlus para uso 
 * dentro da {@code Rede Neural}.
 */
public class SoftPlus extends Ativacao{

   /**
    * Instancia a função de ativação SoftPlus.
    */
   public SoftPlus(){

   }
   @Override
   public void calcular(CamadaDensa camada){
      for(int i = 0; i < camada.saida.lin; i++){
         for(int j = 0; j < camada.saida.col; j++){
            double s = Math.log(1 + Math.exp(camada.somatorio.dado(i, j)));
            camada.saida.editar(i, j, s);
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      for(int i = 0; i < camada.derivada.lin; i++){
         for(int j = 0; j < camada.derivada.col; j++){
            double exp = Math.exp(camada.somatorio.dado(i, j));
            camada.derivada.editar(i, j, (exp / (1 + exp)));
         }
      }
   }
}
