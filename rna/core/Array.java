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

   /**
    * 
    * @param a
    * @param b
    * @param r
    */
   public static void add(double[] a, double[] b, double[] r){
      if(a.length != b.length){
         throw new IllegalArgumentException(
            "As dimensões de A (" + a.length + 
            ") e B (" + b.length +
            ") devem ser iguais."
         );
      }

      if(a.length != r.length){
         throw new IllegalArgumentException(
            "As dimensões de R (" + r.length + 
            ") devem ser iguais as de A e B (" + a.length + ")."
         );      
      }

      for(int i = 0; i < r.length; i++){
         r[i] = a[i] + b[i];
      }
   }

   /**
    * 
    * @param a
    * @param b
    * @param r
    */
   public static void sub(double[] a, double[] b, double[] r){
      if(a.length != b.length){
         throw new IllegalArgumentException(
            "As dimensões de A (" + a.length + 
            ") e B (" + b.length +
            ") devem ser iguais."
         );
      }

      if(a.length != r.length){
         throw new IllegalArgumentException(
            "As dimensões de R (" + r.length + 
            ") devem ser iguais as de A e B (" + a.length + ")."
         );      
      }

      for(int i = 0; i < r.length; i++){
         r[i] = a[i] - b[i];
      }
   }

   /**
    * 
    * @param a
    * @param b
    * @param r
    */
   public static void mult(double[] a, double[] b, double[] r){
      if(a.length != b.length){
         throw new IllegalArgumentException(
            "As dimensões de A (" + a.length + 
            ") e B (" + b.length +
            ") devem ser iguais."
         );
      }

      if(a.length != r.length){
         throw new IllegalArgumentException(
            "As dimensões de R (" + r.length + 
            ") devem ser iguais as de A e B (" + a.length + ")."
         );      
      }

      for(int i = 0; i < r.length; i++){
         r[i] = a[i] * b[i];
      }
   }

   /**
    * 
    * @param a
    * @param e
    * @param r
    */
   public static void escalar(double[] a, double e, double[] r){
      if(a.length != r.length){
         throw new IllegalArgumentException(
            "As linhas de A (" + a.length + 
            ") e R (" + r.length + 
            ") devem ser iguais."
         );
      }

      for(int i = 0; i < r.length; i++){
         r[i] = a[i] * e;
      }     
   }

   public static double produtoEscalar(double[] a, double[] b){
      if(a.length != b.length){
         throw new IllegalArgumentException(
            "As linhas de A (" + a.length + 
            ") e B (" + b.length + 
            ") devem ser iguais."
         );
      }

      double produto = 0;
      for(int i = 0; i < a.length; i++){
         produto += a[i] * b[i];
      }

      return produto;
   }
}
