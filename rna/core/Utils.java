package rna.core;

/**
 * Utilitário geral para a biblioteca.
 */
public class Utils{

   /**
    * Utilitário geral para a biblioteca.
    */
   public Utils(){}

   /**
    * Verifica se o conteúdo do array contém valores maiores que zero.
    * @param arr array base.
    * @return resultado da verificação.
    */
   public boolean apenasMaiorZero(int[] arr){
      for(int i = 0; i < arr.length; i++){
         if(arr[i] < 1) return false;
      }
      return true;
   }

   /**
    * Desserializa o array no array de matrizes de destino.
    * @param arr array contendo os dados.
    * @param dest destino da cópia
    */
   public void copiar(double[] arr, Mat[] dest){
      if(arr.length != (dest.length * dest[0].tamanho())){
         throw new IllegalArgumentException(
            "Tamanhos incompatíveis entre o array (" + arr.length + 
            ") e o destino (" + (dest.length * dest[0].tamanho()) + ")."
         );
      }
   
      int id = 0, i, j, k;
      for(i = 0; i < dest.length; i++){
         for(j = 0; j < dest[i].lin(); j++){
            for(k = 0; k < dest[i].col(); k++){
               dest[i].editar(j, k, arr[id++]);
            }
         }
      }
   }

   /**
    * Desserializa o array na matriz de matriz de destino.
    * @param arr array contendo os dados.
    * @param dest destino da cópia
    */
   public void copiar(double[] arr, Mat[][] dest){
      if(arr.length != (dest.length * dest[0].length * dest[0][0].tamanho())){
         throw new IllegalArgumentException(
            "Tamanhos incompatíveis entre o array (" + arr.length + 
            ") e o destino (" + (dest.length * dest[0].length * dest[0][0].tamanho()) + ")."
         );
      }

      int id = 0;
      int i, j, k, l;
      for(i = 0; i < dest.length; i++){
         for(j = 0; j < dest[i].length; j++){
            for(k = 0; k < dest[i][j].lin(); k++){
               for(l = 0; l < dest[i][j].col(); l++){
                  dest[i][j].editar(k, l, arr[id++]);
               }
            }
         }
      }
   }

   /**
    * Copia o conteúdo contido do array de matrizes para o
    * destino desejado.
    * @param arr array de matrizes contendo os dados.
    * @param dest destino da cópia.
    */
   public void copiar(double[][][] arr, Mat[] dest){
      if((arr.length * arr[0].length * arr[0][0].length) != (dest.length * dest[0].tamanho())){
         throw new IllegalArgumentException(
            "Tamanhos incompatíveis entre o tensor (" + (arr.length * arr[0].length) + 
            ") e o destino (" + (dest.length * dest[0].tamanho()) + ")."
         );
      }

      for(int i = 0; i < dest.length; i++){
         dest[i].copiar(arr[i]);
      }
   }

   /**
    * Copia o conteúdo contido do array de matrizes para o
    * destino desejado.
    * @param arr array de matrizes contendo os dados.
    * @param dest destino da cópia.
    */
   public void copiar(Mat[] arr, Mat[] dest){
      for(int i = 0; i < dest.length; i++){
         dest[i].copiar(arr[i]);
      }
   }
}
