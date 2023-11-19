package rna.core;

import java.util.Random;

public class Matriz{
   static Random random = new Random();

   static int numThreads = 2;

   public static void randomizar(double[][] a){
      int alcance = 2;
      for(int i = 0; i < a.length; i++){
         for(int j = 0; j < a[0].length; j++){
            a[i][j] = random.nextDouble(-alcance, alcance);
         }
      }     
   }

   public static void Xavier(double[][] m){
      int numRows = m.length;
      int numCols = m[0].length;
      
      double limit = Math.sqrt(6.0 / (numRows + numCols));
      
      for (int i = 0; i < numRows; i++) {
         for (int j = 0; j < numCols; j++) {
            m[i][j] = Math.random() * 2 * limit - limit;
         }
      }
   }

   public static void he(double[][] m){
      double desvio = Math.sqrt(2 / m.length);

      for(int i = 0; i < m.length; i++){
         for(int j = 0; j < m[0].length; j++){
            m[i][j] = random.nextGaussian() * desvio;
         }
      }     
   }

   public static double[][] arrayParaMatrizColuna(double[] entrada){
      double[][] matriz = new double[1][entrada.length];
      System.arraycopy(entrada, 0, matriz[0], 0, entrada.length);
      return matriz;
   }

   public static double[][] arrayParaMatrizLinha(double[] entrada){
      double[][] matriz = new double[entrada.length][1];
      
      for(int i = 0; i < entrada.length; i++){
         matriz[i][0] = entrada[i];
      }

      return matriz;
   }

   public static double[] matrizParaArrayColuna(double[][] entrada){
      double[] e = new double[entrada[0].length];
      System.arraycopy(entrada[0], 0, e, 0, e.length);
      return e;
   }

   public static void copiar(double[][] m, double[][] r){
      if(m.length != r.length){
         throw new IllegalArgumentException(
            "As linhas de M (" + m.length + 
            ") e R (" + r.length + 
            ") devem ser iguais"
         );
      }
      if(m[0].length != r[0].length){
         throw new IllegalArgumentException(
            "As colunas de M (" + m[0].length + 
            ") e R (" + r[0].length + 
            ") devem ser iguais"
         );
      }

      for(int i = 0; i < m.length; i++){
         for(int j = 0; j < m[i].length; j++){
            r[i][j] = m[i][j];
         }
      }
   }

   public static double[][] transpor(double[][] a){
      double[][] t = new double[a[0].length][a.length];

      for(int i = 0; i < a.length; i++){
         for(int j = 0; j < a[i].length; j++){
            t[j][i] = a[i][j];
         }
      }

      return t;
   }

   /**
    * Multiplicação matricial convencional seguindo a expressão:
    * <pre>
    * R = A * B
    * </pre>
    * @param a primeita matriz.
    * @param b segunda matriz.
    * @param r matriz contendo o resultado.
    */
   public static void mult(double[][] a, double[][] b, double[][] r){
      if(a[0].length != b.length){
         throw new IllegalArgumentException("Dimensões de A e B incompatíveis");
      }
      if(r.length != a.length){
         throw new IllegalArgumentException(
            "As linhas de A (" + a.length + 
            ") e R (" + r.length + 
            ") devem ser iguais."
         );
      }
      if(r[0].length != b[0].length){
         throw new IllegalArgumentException(
            "As colunas de B (" + b[0].length + 
            ") e R (" + r[0].length + 
            ") devem ser iguais."
         );
      }

      for(int i = 0; i < r.length; i++){
         for(int j = 0; j < r[i].length; j++){
            r[i][j] = 0;
            for(int k = 0; k < a[0].length; k++){
               r[i][j] += a[i][k] * b[k][j];
            }
         }
      }
   }

   /**
    * Multiplicação matricial em paralelo seguindo a expressão:
    * <pre>
    * R = A * B
    * </pre>
    * @param a primeita matriz.
    * @param b segunda matriz.
    * @param r matriz contendo o resultado.
    */
   public static void multT(double[][] a, double[][] b, double[][] r){
      int linA = a.length;
      int colA = a[0].length;
      int linB = b.length;
      int colB = b[0].length;
      int linR = r.length;
      int colR = r[0].length;

      if(colA != linB){
         throw new IllegalArgumentException("Dimensões de A e B incompatíveis");
      }
      if(linR != linA){
         throw new IllegalArgumentException(
            "As linhas de A (" + linA + 
            ") e R (" + linR + 
            ") devem ser iguais."
         );
      }
      if(colR != colB){
         throw new IllegalArgumentException(
            "As colunas de B (" + colB + 
            ") e R (" + colR + 
            ") devem ser iguais."
         );
      }

      int linPorThread = linA / numThreads;
      Thread[] threads = new Thread[numThreads];

      for(int t = 0; t < numThreads; t++){
         final int id = t;

         threads[t] = new Thread(() -> {
            int inicio = id * linPorThread;
            int fim = (id == numThreads - 1) ? linA : (id + 1) * linPorThread;

            for(int i = inicio; i < fim; i++){
               
               for(int j = 0; j < linR; j++){
                  double res = 0;
                  for(int k = 0; k < colA; k++){
                     res += a[i][k] * b[k][j];
                  }
                  
                  synchronized(r[i]){
                     r[i][j] = res;
                  }
               }
            }
         });
         
         threads[t].start();
      }
   
      try{
         for(int i = 0; i < numThreads; i++){
            threads[i].join();
         }
      }catch(InterruptedException e){
         e.printStackTrace();
         System.exit(1);
      }
   }

   /**
    * 
    * @param a
    * @param b
    * @param r
    */
   public static void add(double[][] a, double[][] b, double[][] r){
      if(a.length != b.length){
         throw new IllegalArgumentException("Linhas de A e B são diferentes.");
      }
      if(a[0].length != b[0].length){
         throw new IllegalArgumentException("Colunas de A e B são diferentes.");
      }
      if(a.length != r.length){
         throw new IllegalArgumentException("Linhas de R são diferentes.");
      }
      if(a[0].length != r[0].length){
         throw new IllegalArgumentException("Colunas de R são diferentes.");
      }

      for(int i = 0; i < r.length; i++){
         for(int j = 0; j < r[0].length; j++){
            r[i][j] = a[i][j] + b[i][j];
         }
      }
   }

   public static void sub(double[][] a, double[][] b, double[][] r){
      if(a.length != b.length){
         throw new IllegalArgumentException(
            "Linhas de A (" + a.length + ") e B (" + b.length + ") são diferentes."
         );
      }
      if(a[0].length != b[0].length){
         throw new IllegalArgumentException(
            "Colunas de A (" + a[0].length + ") e B (" + b[0].length + ") são diferentes."
         );
      }
      if(a.length != r.length){
         throw new IllegalArgumentException("Linhas de R são diferentes.");
      }
      if(a[0].length != r[0].length){
         throw new IllegalArgumentException("Colunas de R são diferentes.");
      }
      
      for(int i = 0; i < r.length; i++){
         for(int j = 0; j < r[0].length; j++){
            r[i][j] = a[i][j] - b[i][j];
         }
      }
   }

   public static void hadamard(double[][] a, double[][]b, double[][] r){
      if(a.length != b.length){
         throw new IllegalArgumentException("Linhas de A e B são diferentes.");
      }
      if(a[0].length != b[0].length){
         throw new IllegalArgumentException("Colunas de A e B são diferentes.");
      }
      if(a.length != r.length){
         throw new IllegalArgumentException("Linhas de R são diferentes.");
      }
      if(a[0].length != r[0].length){
         throw new IllegalArgumentException("Colunas de R são diferentes.");
      }

      for(int i = 0; i < r.length; i++){
         for(int j = 0; j < r[i].length; j++){
            r[i][j] = a[i][j] * b[i][j];
         }
      }
   }

   public static void escalar(double[][] a, double e){
      for(int i = 0; i < a.length; i++){
         for(int j = 0; j < a[0].length; j++){
            a[i][j] = a[i][j] * e;
         }
      }     
   }
}
