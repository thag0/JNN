package jnn.ativacoes;

import jnn.core.Tensor4D;

/**
 * Implementação da função de ativação Argmax para uso 
 * dentro dos modelos.
 */
public class Argmax extends Ativacao {

   /**
    * Intancia uma nova função de ativação Softmax.
    * <p>
    *    A função argmax encontra o maior valor de saída dentre os neurônios
    *    da camada e converte ele para 1, as demais saídas dos neurônios serão
    *    convertidas para zero, fazendo a camada classificar uma única saída com
    *    base no maior valor.
    * </p>
    * <p>
    *    A ativação atua em cada linha do tensor individualmente
    * </p>
    * Exemplo:
    * <pre>
    * tensor = [[[
    *    [1, 2, 3], 
    *    [2, 3, 1], 
    *    [3, 1, 2], 
    *]]]
    *
    *argmax.calcular(tensor, tensor);
    *
    *tensor = [[[
    *    [0, 0, 1], 
    *    [0, 1, 0], 
    *    [1, 0, 0], 
    *]]]
    * </pre>
    */
   public Argmax() {}

   @Override
   public void forward(Tensor4D entrada, Tensor4D saida) {
      int canais = entrada.dim1();
      int profundidade = entrada.dim2();
      int linhas = entrada.dim3();
      int colunas = entrada.dim4();
      int maxId;
      double maxVal;
   
      for (int can = 0; can < canais; can++) {
         for (int prof = 0; prof < profundidade; prof++) {
            for (int lin = 0; lin < linhas; lin++) {
               maxId = 0;
               maxVal = entrada.get(can, prof, lin, 0);

               for (int col = 1; col < colunas; col++) {
                  if (entrada.get(can, prof, lin, col) > maxVal) {
                     maxId = col;
                     maxVal = entrada.get(can, prof, lin, col);
                  }
               }
         
               for (int col = 0; col < colunas; col++) {
                  saida.set(((col == maxId) ? 1.0d : 0.0d), can, prof, lin, col);
               }
            }
         }
      }
   }
}
