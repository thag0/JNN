package rna.core;

public class Matriz{

   /**
    * Impelementações de operações matriciais.
    */
   public Matriz(){

   }

   public double[][] arrayParaMatrizColuna(double[] entrada){
      double[][] matriz = new double[1][entrada.length];
      System.arraycopy(entrada, 0, matriz[0], 0, entrada.length);
      return matriz;
   }

   public double[][] arrayParaMatrizLinha(double[] entrada){
      double[][] matriz = new double[entrada.length][1];
      
      for(int i = 0; i < entrada.length; i++){
         matriz[i][0] = entrada[i];
      }

      return matriz;
   }

   public double[] matrizParaArrayColuna(double[][] entrada){
      double[] e = new double[entrada[0].length];
      System.arraycopy(entrada[0], 0, e, 0, e.length);
      return e;
   }

   public void copiar(double[][] m, double[][] r){
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

   /**
    * Substitui cada elemento da matriz pelo valor fornecido.
    * @param m matriz.
    * @param val valor desejado para preenchimento.
    */
   public void preencher(double[][] m, double val){
      for(int i = 0; i < m.length; i++){
         for(int j = 0; j < m[i].length; j++){
            m[i][j] = val;
         }
      }    
   }

   /**
    * Transpõe a matriz fornecida, invertendo suas linhas e colunas.
    * @param m matriz.
    * @return transposta da matriz alvo.
    */
   public double[][] transpor(double[][] m){
      double[][] t = new double[m[0].length][m.length];

      for(int i = 0; i < m.length; i++){
         for(int j = 0; j < m[i].length; j++){
            t[j][i] = m[i][j];
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
   public void mult(double[][] a, double[][] b, double[][] r){
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
            double res = 0;
            for(int k = 0; k < a[0].length; k++){
               res += a[i][k] * b[k][j];
            }
            r[i][j] = res;    
         }
      }
   }

   /**
    * Adiciona o conteúdo resultante da soma entre A e B na matriz R de acordo
    * com a expressão:
    * <pre>
    * R = A + B
    * </pre>
    * @param a primeita matriz.
    * @param b segunda matriz.
    * @param r matriz contendo o resultado da soma.
    */
   public void add(double[][] a, double[][] b, double[][] r){
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

      if(a.length == 1){
         Array.add(a[0], b[0], r[0]);

      }else{
         for(int i = 0; i < r.length; i++){
            for(int j = 0; j < r[0].length; j++){
               r[i][j] = a[i][j] + b[i][j];
            }
         }
      }

   }

   /**
    * Adiciona o conteúdo resultante da subtração entre A e B na matriz R de acordo
    * com a expressão:
    * <pre>
    * R = A - B
    * </pre>
    * @param a primeita matriz.
    * @param b segunda matriz.
    * @param r matriz contendo o resultado da subtração.
    */
   public void sub(double[][] a, double[][] b, double[][] r){
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

      if(a.length == 1){
         Array.sub(a[0], b[0], r[0]);

      }else{
         for(int i = 0; i < r.length; i++){
            for(int j = 0; j < r[0].length; j++){
               r[i][j] = a[i][j] - b[i][j];
            }
         }
      }
      
   }

   /**
    * Adiciona o conteúdo resultante do produto elemeto a elemento entre A e B na matriz 
    * R de acordo com a expressão:
    * <pre>
    * R = A ⊙ B
    * </pre>
    * @param a primeita matriz.
    * @param b segunda matriz.
    * @param r matriz contendo o resultado do produto hadamard.
    */
   public void hadamard(double[][] a, double[][]b, double[][] r){
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

      if(a.length == 1){
         Array.mult(a[0], b[0], r[0]);

      }else{
         for(int i = 0; i < r.length; i++){
            for(int j = 0; j < r[i].length; j++){
               r[i][j] = a[i][j] * b[i][j];
            }
         }
      }

   }

   /**
    * Adiciona o conteúdo resultante da multiplicação elemento a elemento do conteúdo da matriz
    * A por um valor escalar de acordo com a expressão:
    * <pre>
    * R = A * esc
    * </pre>
    * @param a matriz alvo.
    * @param e escalar utilizado para a multiplicação.
    * @param r matriz que terá o resultado.
    */
   public void escalar(double[][] a, double e, double[][] r){
      if(a.length != r.length){
         throw new IllegalArgumentException(
            "As linhas de A (" + a.length + 
            ") e R (" + r.length + 
            ") devem ser iguais."
         );
      }
      if(a[0].length != r[0].length){
         throw new IllegalArgumentException(
            "As colunas de A (" + a.length + 
            ") e R (" + r.length + 
            ") devem ser iguais."
         );
      }

      if(a.length == 1){
         Array.escalar(a[0], e, r[0]);
      
      }else{
         for(int i = 0; i < r.length; i++){
            for(int j = 0; j < r[i].length; j++){
               r[i][j] = a[i][j] * e;
            }
         }
      }

   }
}
