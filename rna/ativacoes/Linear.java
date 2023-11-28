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
      int i, j;
      
      for(i = 0; i < camada.saida.lin; i++){
         for(j = 0; j < camada.saida.col; j++){
            camada.saida.editar(i, j, camada.somatorio.dado(i, j));
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      int i, j;
      double grad;

      for(i = 0; i < camada.derivada.lin; i++){
         for(j = 0; j < camada.derivada.col; j++){
            grad = camada.gradienteSaida.dado(i, j);
            camada.derivada.editar(i, j, (1 * grad));
         }
      }
   }
}
