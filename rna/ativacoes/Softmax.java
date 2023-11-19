package rna.ativacoes;

import rna.estrutura.CamadaDensa;

/**
 * Implementação da função de ativação Softmax para uso 
 * dentro da {@code Rede Neural}.
 */
public class Softmax extends Ativacao{

   /**
    * Instancia a função de ativação Softmax.
    * <p>
    * A função Softmax  transforma os valores de entrada em probabilidades normalizadas, 
    * permitindo que o neurônio com a maior saída tenha uma probabilidade mais alta.
    * </p>
    */
   public Softmax(){

   }

   @Override
   public void calcular(CamadaDensa camada){
      double somaExp = 0;

      for(int i = 0; i < camada.somatorio[0].length; i++){
         somaExp += Math.exp(camada.somatorio[0][i]);
      }

      for(int i = 0; i < camada.saida[0].length; i++){
         camada.saida[0][i] = Math.exp(camada.somatorio[0][i]) / somaExp;
         somaExp += Math.exp(camada.somatorio[0][i]);
      }
   }
}
