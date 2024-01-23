package rna.core;

public class OpArray{

   /**
    * Interface para execução de operação utilizando arrays.
    */
   public OpArray(){}

   public void print(double[] arr){
      System.out.println("Array = [");
      System.out.print("  " + arr[0]);
      for(int i = 1; i < arr.length; i++){
         System.out.print(", " + arr[i]);
      }
      System.out.println("]");
   }

   /**
    * Preenche todo o conteúdo do array com o valor fornecido.
    * @param arr array.
    * @param val valor desejado.
    */
   public void preencher(double[] arr, double val){
      int n = arr.length;
      for(int i = 0; i < n; i++){
         arr[i] = val;
      }
   }

   /**
    * Adiciona o conteúdo resultante da soma entre A e B no array R de acordo
    * com a expressão:
    * <pre>
    * R = A + B
    * </pre>
    * @param a primeiro array.
    * @param b segundo array.
    * @param r array contendo o resultado da soma.
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
      int n = r.length;
      for(int i = 0; i < n; i++){
         r[i] += b[i];
      }
   }

   /**
    * Adiciona o conteúdo resultante da subtração entre A e B no array R de acordo
    * com a expressão:
    * <pre>
    * R = A - B
    * </pre>
    * @param a primeiro array.
    * @param b segundo array.
    * @param r array contendo o resultado da subtração.
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
      int n = r.length;
      for(int i = 0; i < n; i++){
         r[i] -= b[i];
      }
   }

   /**
    * Adiciona o conteúdo resultante da multiplicação entre A e B no array R de acordo
    * com a expressão:
    * <pre>
    * R = A * B
    * </pre>
    * @param a primeiro array.
    * @param b segundo array.
    * @param r array contendo o resultado da multiplicação.
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

      System.arraycopy(a, 0, r, 0, r.length);
      int n = r.length;
      for(int i = 0; i < n; i++){
         r[i] *= b[i];
      }
   }

   /**
    * Aidiciona o conteúdo resultado da divisão de cada elemento do array A pelo
    * valor escalar fornecido, de acordo com a expressão:
    * <pre>
    * R = A / e
    * </pre>
    * @param a array base.
    * @param e valor usado para dividir os elementos dos array.
    * @param r array contendo o resultado da divisão.
    */
   public void divEscalar(double[] a, double e, double[] r){
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
    * Aidiciona o conteúdo resultado da multiplicação de cada elemento do array A pelo
    * valor escalar fornecido, de acordo com a expressão:
    * <pre>
    * R = A * e
    * </pre>
    * @param a array base.
    * @param e valor usado para multiplicar os elementos dos array.
    * @param r array contendo o resultado da divisão.
    */
   public void multEscalar(double[] a, double e, double[] r){
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

   /**
    * Calcula o resultado do produto escalar entre os arrays A e B.
    * @param a primeiro array.
    * @param b segundo array.
    * @return resultado do produto escalar entre A e B.
    */
   public double produtoEscalar(double[] a, double[] b){
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
