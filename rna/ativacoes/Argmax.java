package rna.ativacoes;

import rna.estrutura.CamadaDensa;

/**
 * Implementação da função de ativação Argmax para uso 
 * dentro da {@code Rede Neural}.
 */
public class Argmax extends Ativacao{

   /**
    * Intancia uma nova função de ativação Softmax.
    * <p>
    *    A função argmax encontra o maior valor de saída dentre os neurônios
    *    da camada e converte ele para 1, as demais saídas dos neurônios serão
    *    convertidas para zero, fazendo a camada classificar uma única saída com
    *    base no maior valor.
    * </p>
    */
   public Argmax(){

   }

   @Override
   public void calcular(CamadaDensa camada){
      int indiceMaximo = 0;
      double valorMaximo = camada.somatorio[0][0];

      for(int i = 1; i < camada.somatorio[0].length; i++){
         if(camada.somatorio[0][i] > valorMaximo){
            indiceMaximo = i;
            valorMaximo = camada.somatorio[0][i];
         }
      }

      for(int i = 0; i < camada.somatorio[0].length; i++){
         camada.saida[0][i] = (i == indiceMaximo) ? 1 : 0;
      }
   }
}
