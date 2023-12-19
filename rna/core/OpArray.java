package rna.core;

public class OpArray{

   public void preencher(double[] arr, double val){
      for(int i = 0; i < arr.length; i++){
         arr[i] = val;
      }
   }

   /**
    * 
    * @param a
    * @param b
    * @param r
    */
   public void add(double[] a, double[] b, double[] r){
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

      System.arraycopy(a, 0, r, 0, r.length);
      for(int i = 0; i < r.length; i++){
         r[i] += b[i];
      }
   }

   /**
    * 
    * @param a
    * @param b
    * @param r
    */
   public void sub(double[] a, double[] b, double[] r){
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

      System.arraycopy(a, 0, r, 0, r.length);
      for(int i = 0; i < r.length; i++){
         r[i] -= b[i];
      }
   }

   /**
    * 
    * @param a
    * @param b
    * @param r
    */
   public void mult(double[] a, double[] b, double[] r){
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
   public void dividirEscalar(double[] a, double e, double[] r){
      if(a.length != r.length){
         throw new IllegalArgumentException(
            "As dimensões de R (" + r.length + 
            ") devem ser iguais as de A e B (" + a.length + ")."
         );      
      }

      for(int i = 0; i < r.length; i++){
         r[i] = a[i] / e;
      }
   }

   /**
    * 
    * @param a
    * @param e
    * @param r
    */
   public void escalar(double[] a, double e, double[] r){
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

   public  double produtoEscalar(double[] a, double[] b){
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
