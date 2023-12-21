package rna.core;

public class Utils{

   public boolean apenasMaiorZero(int[] arr){
      for(int i : arr){
         if(i < 1) return false;
      }
      return true;
   }

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

   public void copiar(double[][][] tensor, Mat[] dest){
      if((tensor.length * tensor[0].length * tensor[0][0].length) != (dest.length * dest[0].tamanho())){
         throw new IllegalArgumentException(
            "Tamanhos incompatíveis entre o tensor (" + (tensor.length * tensor[0].length) + 
            ") e o destino (" + (dest.length * dest[0].tamanho()) + ")."
         );
      }

      for(int i = 0; i < dest.length; i++){
         dest[i].copiar(tensor[i]);
      }
   }

   public void copiar(Mat[] arrMat, Mat[] dest){
      for(int i = 0; i < dest.length; i++){
         dest[i].copiar(arrMat[i]);
      }
   }
}
