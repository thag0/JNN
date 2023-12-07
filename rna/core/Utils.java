package rna.core;

public class Utils{

   public void copiar(double[] arr, Mat[] dest){
      int id = 0;
      for(int i = 0; i < dest.length; i++){
         for(int j = 0; j < dest[i].lin; j++){
            for(int k = 0; k < dest[i].col; k++){
               dest[i].editar(j, k, arr[id++]);
            }
         }
      }
   }

   public void copiar(double[][][] tensor, Mat[] dest){
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
