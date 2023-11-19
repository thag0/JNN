package rna.core;

public class Array{

   /**
    * Copia o conteúdo de A para R.
    * @param a array contendo os dados.
    * @param r array onde serão copiado os dados de A.
    * @throws IllegalArgumentException se as dimensões de A e R forem diferentes.
    */
   public static void copiar(double[] a, double[] r){
      if(a.length != r.length){
         throw new IllegalArgumentException(
            "Dimensões de A (" + a.length + 
            ") e R (" + r.length + 
            ") são diferentes."
         );
      }

      for(int i = 0; i < r.length; i++){
         r[i] = a[i];
      }
   }
}
