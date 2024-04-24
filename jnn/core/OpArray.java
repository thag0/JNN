package jnn.core;

public class OpArray {

   /**
    * Interface para execução de operação utilizando arrays.
    */
   public OpArray() {}

   /**
    * Exibe o conteúdo do array.
    * @param arr array desejado.
    */
   public void print(double[] arr) {
      System.out.println("Array = [");
      System.out.print("  " + arr[0]);
      for (int i = 1; i < arr.length; i++) {
         System.out.print(", " + arr[i]);
      }
      System.out.println("]");
   }

   /**
    * Preenche todo o conteúdo do array com o valor fornecido.
    * @param arr array.
    * @param val valor desejado.
    */
   public void preencher(double[] arr, double val) {
      int n = arr.length;
      for (int i = 0; i < n; i++) {
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
    * @param dest array contendo o resultado da soma.
    */
   public void add(double[] a, double[] b, double[] dest) {
      if (a.length != b.length) {
         throw new IllegalArgumentException(
            "As dimensões de A (" + a.length + 
            ") e B (" + b.length +
            ") devem ser iguais."
         );
      }

      if (a.length != dest.length) {
         throw new IllegalArgumentException(
            "As dimensões de R (" + dest.length + 
            ") devem ser iguais as de A e B (" + a.length + ")."
         );      
      }

      System.arraycopy(a, 0, dest, 0, dest.length);
      int n = dest.length;
      for (int i = 0; i < n; i++) {
         dest[i] += b[i];
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
    * @param dest array contendo o resultado da subtração.
    */
   public void sub(double[] a, double[] b, double[] dest) {
      if (a.length != b.length) {
         throw new IllegalArgumentException(
            "As dimensões de A (" + a.length + 
            ") e B (" + b.length +
            ") devem ser iguais."
         );
      }

      if (a.length != dest.length) {
         throw new IllegalArgumentException(
            "As dimensões de R (" + dest.length + 
            ") devem ser iguais as de A e B (" + a.length + ")."
         );      
      }

      System.arraycopy(a, 0, dest, 0, dest.length);
      int n = dest.length;
      for (int i = 0; i < n; i++) {
         dest[i] -= b[i];
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
    * @param dest array contendo o resultado da multiplicação.
    */
   public void mult(double[] a, double[] b, double[] dest) {
      if (a.length != b.length) {
         throw new IllegalArgumentException(
            "As dimensões de A (" + a.length + 
            ") e B (" + b.length +
            ") devem ser iguais."
         );
      }

      if (a.length != dest.length) {
         throw new IllegalArgumentException(
            "As dimensões de R (" + dest.length + 
            ") devem ser iguais as de A e B (" + a.length + ")."
         );      
      }

      System.arraycopy(a, 0, dest, 0, dest.length);
      int n = dest.length;
      for (int i = 0; i < n; i++) {
         dest[i] *= b[i];
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
    * @param dest array contendo o resultado da divisão.
    */
   public void divEscalar(double[] a, double e, double[] dest) {
      if (a.length != dest.length) {
         throw new IllegalArgumentException(
            "As dimensões de R (" + dest.length + 
            ") devem ser iguais as de A e B (" + a.length + ")."
         );      
      }

      for (int i = 0; i < dest.length; i++) {
         dest[i] = a[i] / e;
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
    * @param dest array contendo o resultado da divisão.
    */
   public void multEscalar(double[] a, double e, double[] dest) {
      if (a.length != dest.length) {
         throw new IllegalArgumentException(
            "As linhas de A (" + a.length + 
            ") e R (" + dest.length + 
            ") devem ser iguais."
         );
      }

      for (int i = 0; i < dest.length; i++) {
         dest[i] = a[i] * e;
      }     
   }

   /**
    * Calcula o resultado do produto escalar entre os arrays A e B.
    * @param a primeiro array.
    * @param b segundo array.
    * @return resultado do produto escalar entre A e B.
    */
   public double produtoEscalar(double[] a, double[] b) {
      if (a.length != b.length) {
         throw new IllegalArgumentException(
            "As linhas de A (" + a.length + 
            ") e B (" + b.length + 
            ") devem ser iguais."
         );
      }

      double prod = 0;
      for (int i = 0; i < a.length; i++) {
         prod += a[i] * b[i];
      }

      return prod;
   }

   /**
    * Inverte todo o conteúdo do array
    * @param arr array base;
    */
   public void inverter(double[] arr) {
      int inicio = 0;
      int fim = arr.length - 1;
      double temp;
      
      while (inicio < fim) {
         temp = arr[inicio];
         arr[inicio] = arr[fim];
         arr[fim] = temp;
         inicio++;
         fim--;
      }
   }
}
