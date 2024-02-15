package rna.ativacoes;

import rna.camadas.Densa;

/**
 * Implementação da função de ativação Argmax para uso 
 * dentro dos modelos.
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
   public void calcular(Densa camada){
      int indiceMaximo = 0;
      double valorMaximo = camada.somatorio.elemento(0, 0, 0, 0);

      int colunas = camada.somatorio.dim4();
      for(int i = 1; i < colunas; i++){
         if(camada.somatorio.elemento(0, 0, 0, i) > valorMaximo){
            indiceMaximo = i;
            valorMaximo = camada.somatorio.elemento(0, 0, 0, i);
         }
      }

      for(int i = 0; i < colunas; i++){
         camada.saida.editar(0, 0, 0, i, ((i == indiceMaximo) ? 1 : 0));
      }
   }
}
