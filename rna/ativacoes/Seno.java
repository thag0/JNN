package rna.ativacoes;

import rna.estrutura.CamadaDensa;

/**
 * Implementação da função de ativação Seno para uso dentro 
 * da {@code Rede Neural}.
 */
public class Seno extends Ativacao{

   /**
    * Instancia a função de ativação Seno.
    */
   public Seno(){

   }

   @Override
   public void calcular(CamadaDensa camada){
      for(int i = 0; i < camada.saida.lin; i++){
         for(int j = 0; j < camada.saida.col; j++){
            camada.saida.editar(i, j, Math.sin(camada.somatorio.dado(i, j)));
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
            d = Math.cos(camada.somatorio.dado(i, j));
            camada.derivada.editar(i, j, (grad * d));
         }
      }
   }
   
}
