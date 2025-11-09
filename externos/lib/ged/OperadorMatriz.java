package externos.lib.ged;;


/**
 * Operador de matrizes do Ged.
 */
class OperadorMatriz {


   /**
    * Contém implementações de operações matriciais para dados
    * int, float e double.
    */
   public OperadorMatriz() {}

   public Object paraArray(Object matriz) {
      if (matriz instanceof int[][]) {
         int[][] m = (int[][]) matriz;
         return vetorizar(m);         
      
      } else if (matriz instanceof float[][]) {
         float[][] m = (float[][]) matriz;
         return vetorizar(m);

      } else if (matriz instanceof double[][]) {
         double[][] m = (double[][]) matriz;
         return vetorizar(m);

      } else {
         throw new IllegalArgumentException(
            "Tipo de dado (" + matriz.getClass().getSimpleName() +") não suportado."
         );
      }
   }

   private int[] vetorizar(int[][] matriz) {
      int[] arr = new int[matriz.length * matriz[0].length];
   
      int cont = 0;
      for (int i = 0; i < matriz.length; i++) {
         for (int j = 0; j < matriz[i].length; j++) {
            arr[cont] = matriz[i][j];
            cont++;
         }
      }

      return arr;
   }

   private float[] vetorizar(float[][] matriz) {
      float[] arr = new float[matriz.length * matriz[0].length];
   
      int cont = 0;
      for (int i = 0; i < matriz.length; i++) {
         for (int j = 0; j < matriz[i].length; j++) {
            arr[cont] = matriz[i][j];
            cont++;
         }
      }

      return arr;
   }

   private double[] vetorizar(double[][] matriz) {
      double[] arr = new double[matriz.length * matriz[0].length];
   
      int cont = 0;
      for (int i = 0; i < matriz.length; i++) {
         for (int j = 0; j < matriz[i].length; j++) {
            arr[cont] = matriz[i][j];
            cont++;
         }
      }

      return arr;
   }

   //sublinhas

   public Object obterSubLins(Object matriz, int inicio, int fim) {
      if (matriz instanceof int[][]) {
         int[][] m = (int[][]) matriz;
         return obterSubLinhas(m, inicio, fim);    
      
      } else if (matriz instanceof float[][]) {
         float[][] m = (float[][]) matriz;
         return obterSubLinhas(m, inicio, fim);   

      } else if (matriz instanceof double[][]) {
         double[][] m = (double[][]) matriz;
         return obterSubLinhas(m, inicio, fim);   

      } else {
         throw new IllegalArgumentException(
            "Tipo de dado (" + matriz.getClass().getSimpleName() +") não suportado."
         );
      }   
   }
   
   private int[][] obterSubLinhas(int[][] dados, int inicio, int fim) {
      if (inicio < 0 || fim > dados.length || inicio >= fim) {
         throw new IllegalArgumentException("Índices de início ou fim inválidos.");
      }

      int linhas = fim - inicio;
      int colunas = dados[0].length;
      int[][] subMatriz = new int[linhas][colunas];

      for (int i = 0; i < linhas; i++) {
         System.arraycopy(dados[inicio + i], 0, subMatriz[i], 0, colunas);
      }

      return subMatriz;
   }

   private float[][] obterSubLinhas(float[][] dados, int inicio, int fim) {
      if (inicio < 0 || fim > dados.length || inicio >= fim) {
         throw new IllegalArgumentException("Índices de início ou fim inválidos.");
      }

      int linhas = fim - inicio;
      int colunas = dados[0].length;
      float[][] subMatriz = new float[linhas][colunas];

      for (int i = 0; i < linhas; i++) {
         System.arraycopy(dados[inicio + i], 0, subMatriz[i], 0, colunas);
      }

      return subMatriz;
   }

   private double[][] obterSubLinhas(double[][] dados, int inicio, int fim) {
      if (inicio < 0 || fim > dados.length || inicio >= fim) {
         throw new IllegalArgumentException("Índices de início ou fim inválidos.");
      }

      int linhas = fim - inicio;
      int colunas = dados[0].length;
      double[][] subMatriz = new double[linhas][colunas];

      for (int i = 0; i < linhas; i++) {
         System.arraycopy(dados[inicio + i], 0, subMatriz[i], 0, colunas);
      }

      return subMatriz;
   }

   //subcolunas

   public Object obterSubCols(Object matriz, int inicio, int fim) {
      if (matriz instanceof int[][]) {
         int[][] m = (int[][]) matriz;
         return obterSubColunas(m, inicio, fim);    
      
      } else if (matriz instanceof float[][]) {
         float[][] m = (float[][]) matriz;
         return obterSubColunas(m, inicio, fim);   

      } else if (matriz instanceof double[][]) {
         double[][] m = (double[][]) matriz;
         return obterSubColunas(m, inicio, fim);   

      } else {
         throw new IllegalArgumentException(
            "Tipo de dado (" + matriz.getClass().getSimpleName() +") não suportado."
         );
      }   
   }

   private int[][] obterSubColunas(int[][] matriz, int inicio, int fim) {
      if (inicio < 0 || fim > matriz[0].length || inicio >= fim) {
         throw new IllegalArgumentException("Índices de início ou fim inválidos para colunas.");
      }

      int linhas = matriz.length;
      int colunas = fim - inicio;
      int[][] subColunas = new int[linhas][colunas];

      for (int i = 0; i < linhas; i++) {
         for (int j = 0; j < colunas; j++) {
            System.arraycopy(matriz[i], inicio, subColunas[i], 0, colunas);
         }
      }

      return subColunas;
   }

   private float[][] obterSubColunas(float[][] matriz, int inicio, int fim) {
      if (inicio < 0 || fim > matriz[0].length || inicio >= fim) {
         throw new IllegalArgumentException("Índices de início ou fim inválidos para colunas.");
      }

      int linhas = matriz.length;
      int colunas = fim - inicio;
      float[][] subColunas = new float[linhas][colunas];

      for (int i = 0; i < linhas; i++) {
         for (int j = 0; j < colunas; j++) {
            System.arraycopy(matriz[i], inicio, subColunas[i], 0, colunas);
         }
      }

      return subColunas;
   }

   private double[][] obterSubColunas(double[][] matriz, int inicio, int fim) {
      if (inicio < 0 || fim > matriz[0].length || inicio >= fim) {
         throw new IllegalArgumentException("Índices de início ou fim inválidos para colunas.");
      }

      int linhas = matriz.length;
      int colunas = fim - inicio;
      double[][] subColunas = new double[linhas][colunas];

      for (int i = 0; i < linhas; i++) {
         for (int j = 0; j < colunas; j++) {
            System.arraycopy(matriz[i], inicio, subColunas[i], 0, colunas);
         }
      }

      return subColunas;
   }

   //preencher
   
   public void preencherMatriz(Object matriz, Number valor) {
      if (matriz == null || valor == null) {
         throw new IllegalArgumentException("Parâmetros fornecidos não podem ser nulos.");
      }

      if (matriz instanceof int[][]) {
         int[][] m = (int[][]) matriz;
         int v = valor.intValue();
         preencherMatriz(m, v);
         matriz = (Object) m;
      
      } else if (matriz instanceof float[][]) {
         float[][] m = (float[][]) matriz;
         float v = valor.floatValue();
         preencherMatriz(m, v);
         matriz = (Object) m;
      
      } else if (matriz instanceof double[][]) {
         double[][] m = (double[][]) matriz;
         double v = valor.doubleValue();
         preencherMatriz(m, v);
         matriz = (Object) m;
      
      } else {
         throw new IllegalArgumentException("Tipo de matriz não suportado.");
      }
   }

   private void preencherMatriz(int[][] matriz, int valor) {
      for (int i = 0; i < matriz.length; i++) {
         for (int j = 0; j < matriz[i].length; j++) {
            matriz[i][j] = valor;
         }
      }
   }

   private void preencherMatriz(float[][] matriz, float valor) {
      for (int i = 0; i < matriz.length; i++) {
         for (int j = 0; j < matriz[i].length; j++) {
            matriz[i][j] = valor;
         }
      }
   }

   private void preencherMatriz(double[][] matriz, double valor) {
      for (int i = 0; i < matriz.length; i++) {
         for (int j = 0; j < matriz[i].length; j++) {
            matriz[i][j] = valor;
         }
      }
   }
   

   //identidade

   public void matId(Object matriz) {
      if (matriz == null) {
         throw new IllegalArgumentException("A matriz fornecida não pode ser nula.");
      }

      if (matriz instanceof int[][]) {
         int[][] mat = (int[][]) matriz;
         matrizIdentidade(mat);
         matriz = (Object) mat;
      
      } else if (matriz instanceof float[][]) {
         float[][] mat = (float[][]) matriz;
         matrizIdentidade(mat);
         matriz = (Object) mat;
      
      } else if (matriz instanceof double[][]) {
         double[][] mat = (double[][]) matriz;
         matrizIdentidade(mat);
         matriz = (Object) mat;
      
      } else {
         throw new IllegalArgumentException("Tipo de matriz não suportado.");
      }
   }

   private void matrizIdentidade(int[][] matriz) {
      for (int i = 0; i < matriz.length; i++) {
         for (int j = 0; j < matriz[i].length; j++) {
            matriz[i][j] = (i == j) ? 1 : 0;
         }
      }       
   }

   private void matrizIdentidade(float[][] matriz) {
      for (int i = 0; i < matriz.length; i++) {
         for (int j = 0; j < matriz[i].length; j++) {
            matriz[i][j] = (i == j) ? 1 : 0;
         }
      }       
   }

   private void matrizIdentidade(double[][] matriz) {
      for (int i = 0; i < matriz.length; i++) {
         for (int j = 0; j < matriz[i].length; j++) {
            matriz[i][j] = (i == j) ? 1 : 0;
         }
      }       
   }

   //transposição 

   public Object matTransp(Object matriz) {
      if (matriz instanceof int[][]) {
         int[][] m = (int[][]) matriz;
         return transporMatriz(m);

      } else if (matriz instanceof float[][]) {
         float[][] m = (float[][]) matriz;
         return transporMatriz(m);

      } else if (matriz instanceof double[][]) {
         double[][] m = (double[][]) matriz;
         return transporMatriz(m);

      } else {
         throw new IllegalArgumentException(
            "Tipo de dado (" + matriz.getClass().getSimpleName() +") não suportado."
         );
      }
   }
   
   private int[][] transporMatriz(int[][] matriz) {
      int linhas = matriz.length;
      int colunas = matriz[0].length;
      int[][] transposta = new int[colunas][linhas];
  
      for (int i = 0; i < linhas; i++) {
         for (int j = 0; j < colunas; j++) {
            transposta[j][i] = matriz[i][j];
         }
      }
  
      return transposta;
   }

   private float[][] transporMatriz(float[][] matriz) {
      int linhas = matriz.length;
      int colunas = matriz[0].length;
      float[][] transposta = new float[colunas][linhas];
  
      for (int i = 0; i < linhas; i++) {
         for (int j = 0; j < colunas; j++) {
            transposta[j][i] = matriz[i][j];
         }
      }
  
      return transposta;
   }

   private double[][] transporMatriz(double[][] matriz) {
      int linhas = matriz.length;
      int colunas = matriz[0].length;
      double[][] transposta = new double[colunas][linhas];
  
      for (int i = 0; i < linhas; i++) {
         for (int j = 0; j < colunas; j++) {
            transposta[j][i] = matriz[i][j];
         }
      }
  
      return transposta;
   }

   
   //verificação de dimensionalidade pra soma, subtração, hadamard
   
   private void dimensoesIguais(int[][] a, int[][] b, int[][] r) {
      if (a.length != b.length || a[0].length != b[0].length || a.length != r.length || a[0].length != r[0].length) {
         throw new IllegalArgumentException("As dimensões de A, B e R não são iguais.");
      }
  }

   private void dimensoesIguais(float[][] a, float[][] b, float[][] r) {
      if (a.length != b.length || a[0].length != b[0].length || a.length != r.length || a[0].length != r[0].length) {
         throw new IllegalArgumentException("As dimensões de A, B e R não são iguais.");
      }
  }

   private void dimensoesIguais(double[][] a, double[][] b, double[][] r) {
      if (a.length != b.length || a[0].length != b[0].length || a.length != r.length || a[0].length != r[0].length) {
         throw new IllegalArgumentException("As dimensões de A, B e R não são iguais.");
      }
  }


   //verificação de dimensionalidade pra multiplicação
   
   private void dimensoesIguaisMult(int[][] a, int[][] b, int[][] r) {
      if (a[0].length != b.length) {
          throw new IllegalArgumentException("Dimensões de A e B incompatíveis para multiplicação");
      }
      if (r.length != a.length || r[0].length != b[0].length) {
          throw new IllegalArgumentException("Dimensões de R incompatíveis com o resultado da multiplicação");
      }
   }

   private void dimensoesIguaisMult(float[][] a, float[][] b, float[][] r) {
      if (a[0].length != b.length) {
          throw new IllegalArgumentException("Dimensões de A e B incompatíveis para multiplicação");
      }
      if (r.length != a.length || r[0].length != b[0].length) {
          throw new IllegalArgumentException("Dimensões de R incompatíveis com o resultado da multiplicação");
      }
   }

   private void dimensoesIguaisMult(double[][] a, double[][] b, double[][] r) {
      if (a[0].length != b.length) {
          throw new IllegalArgumentException("Dimensões de A e B incompatíveis para multiplicação");
      }
      if (r.length != a.length || r[0].length != b[0].length) {
          throw new IllegalArgumentException("Dimensões de R incompatíveis com o resultado da multiplicação");
      }
   }

   // OPERAÇÕES MATRICIAIS -------------------------------------
   //soma

   public void matAdd(Object a, Object b, Object r) {
      if (a == null || b == null || r == null) {
         throw new IllegalArgumentException("As matrizes fornecidas não podem ser nulas.");
      }

      if ((a instanceof int[][]) && (b instanceof int[][]) && (r instanceof int[][])) {
         int[][] m1 = (int[][]) a;
         int[][] m2 = (int[][]) b;
         int[][] mr = (int[][]) r;
         somarMatrizes(m1, m2, mr);
         
      } else if ((a instanceof float[][]) && (b instanceof float[][]) && (r instanceof float[][])) {
         float[][] m1 = (float[][]) a;
         float[][] m2 = (float[][]) b;
         float[][] mr = (float[][]) r;
         somarMatrizes(m1, m2, mr);
         
      } else if ((a instanceof double[][]) && (b instanceof double[][]) && (r instanceof double[][])) {
         double[][] m1 = (double[][]) a;
         double[][] m2 = (double[][]) b;
         double[][] mr = (double[][]) r;
         somarMatrizes(m1, m2, mr);
         
      } else {
         throw new IllegalArgumentException("Tipos de matrizes fornecidas não suportados.");
      }
   }

   private void somarMatrizes(int[][] a, int[][] b, int[][] r) {
      dimensoesIguais(a, b, r);

      for (int i = 0; i < r.length; i++) {
         for (int j = 0; j < r[i].length; j++) {
            r[i][j] = a[i][j] + b[i][j];
         }
      }
   }

   private void somarMatrizes(float[][] a, float[][] b, float[][] r) {
      dimensoesIguais(a, b, r);

      for (int i = 0; i < r.length; i++) {
         for (int j = 0; j < r[i].length; j++) {
            r[i][j] = a[i][j] + b[i][j];
         }
      }
   }

   private void somarMatrizes(double[][] a, double[][] b, double[][] r) {
      dimensoesIguais(a, b, r);

      for (int i = 0; i < r.length; i++) {
         for (int j = 0; j < r[i].length; j++) {
            r[i][j] = a[i][j] + b[i][j];
         }
      }
   }

   //subtração

   public void matSub(Object a, Object b, Object r) {
      if (a == null || b == null || r == null) {
         throw new IllegalArgumentException("As matrizes fornecidas não podem ser nulas.");
      }

      if ((a instanceof int[][]) && (b instanceof int[][]) && (r instanceof int[][])) {
         int[][] m1 = (int[][]) a;
         int[][] m2 = (int[][]) b;
         int[][] mr = (int[][]) r;
         subtrairMatrizes(m1, m2, mr);
         
      } else if ((a instanceof float[][]) && (b instanceof float[][]) && (r instanceof float[][])) {
         float[][] m1 = (float[][]) a;
         float[][] m2 = (float[][]) b;
         float[][] mr = (float[][]) r;
         subtrairMatrizes(m1, m2, mr);
         
      } else if ((a instanceof double[][]) && (b instanceof double[][]) && (r instanceof double[][])) {
         double[][] m1 = (double[][]) a;
         double[][] m2 = (double[][]) b;
         double[][] mr = (double[][]) r;
         subtrairMatrizes(m1, m2, mr);
         
      } else {
         throw new IllegalArgumentException("Tipos de matrizes fornecidas não suportados.");
      }
   }

   private void subtrairMatrizes(int[][] a, int[][] b, int[][] r) {
      dimensoesIguais(a, b, r);

      for (int i = 0; i < r.length; i++) {
         for (int j = 0; j < r[i].length; j++) {
            r[i][j] = a[i][j] - b[i][j];
         }
      }
   }

   private void subtrairMatrizes(float[][] a, float[][] b, float[][] r) {
      dimensoesIguais(a, b, r);

      for (int i = 0; i < r.length; i++) {
         for (int j = 0; j < r[i].length; j++) {
            r[i][j] = a[i][j] - b[i][j];
         }
      }
   }

   private void subtrairMatrizes(double[][] a, double[][] b, double[][] r) {
      dimensoesIguais(a, b, r);

      for (int i = 0; i < r.length; i++) {
         for (int j = 0; j < r[i].length; j++) {
            r[i][j] = a[i][j] - b[i][j];
         }
      }
   }


   //multiplicação

   public void matMult(Object a, Object b, Object r) {
      if (a == null || b == null || r == null) {
         throw new IllegalArgumentException("As matrizes fornecidas não podem ser nulas.");
      }

      if ((a instanceof int[][]) && (b instanceof int[][]) && (r instanceof int[][])) {
         int[][] m1 = (int[][]) a;
         int[][] m2 = (int[][]) b;
         int[][] mr = (int[][]) r;
         multiplicarMatrizes(m1, m2, mr);
         
      } else if ((a instanceof float[][]) && (b instanceof float[][]) && (r instanceof float[][])) {
         float[][] m1 = (float[][]) a;
         float[][] m2 = (float[][]) b;
         float[][] mr = (float[][]) r;
         multiplicarMatrizes(m1, m2, mr);
         
      } else if ((a instanceof double[][]) && (b instanceof double[][]) && (r instanceof double[][])) {
         double[][] m1 = (double[][]) a;
         double[][] m2 = (double[][]) b;
         double[][] mr = (double[][]) r;
         multiplicarMatrizes(m1, m2, mr);
         
      } else {
         throw new IllegalArgumentException("Tipos de matrizes fornecidas não suportados.");   
      }
   }

   private void multiplicarMatrizes(int[][] a, int[][] b, int[][] r) {
      dimensoesIguaisMult(a, b, r);

      int tamInterno = a[0].length;

      for (int i = 0; i < r.length; i++) {
         for (int j = 0; j < r[i].length; j++) {

            r[i][j] = 0;
            for (int k = 0; k < tamInterno; k++) {
               r[i][j] += a[i][k] * b[k][j];
            }
         }
      }
   }

   private void multiplicarMatrizes(float[][] a, float[][] b, float[][] r) {
      dimensoesIguaisMult(a, b, r);

      int tamInterno = a[0].length;

      for (int i = 0; i < r.length; i++) {
         for (int j = 0; j < r[i].length; j++) {

            r[i][j] = 0;
            for (int k = 0; k < tamInterno; k++) {
               r[i][j] += a[i][k] * b[k][j];
            }
         }
      }
   }

   private void multiplicarMatrizes(double[][] a, double[][] b, double[][] r) {
      dimensoesIguaisMult(a, b, r);

      int tamInterno = a[0].length;

      for (int i = 0; i < r.length; i++) {
         for (int j = 0; j < r[i].length; j++) {

            r[i][j] = 0;
            for (int k = 0; k < tamInterno; k++) {
               r[i][j] += a[i][k] * b[k][j];
            }
         }
      }
   }

   //escalar

   public void matMultEscalar(Object mat, Number escalar) {
      if (mat == null || escalar == null) {
         throw new IllegalArgumentException("Os parâmetros fornecidos não podem ser nulos.");
      }

      if (mat instanceof int[][]) {
         int[][] m = (int[][]) mat;
         int e = escalar.intValue();
         multilpicarEscalar(m, e);
      
      } else if (mat instanceof float[][]) {
         float[][] m = (float[][]) mat;
         float e = escalar.floatValue();
         multilpicarEscalar(m, e);

      } else if (mat instanceof double[][]) {
         double[][] m = (double[][]) mat;
         double e = escalar.doubleValue();
         multilpicarEscalar(m, e);
      
      } else {
         throw new IllegalArgumentException("Tipo de matriz não suportado.");
      }
   }

   private void multilpicarEscalar(int[][] matriz, int escalar) {
      for (int i = 0; i < matriz.length; i++) {
         for (int j = 0; j < matriz[i].length; j++) {
            matriz[i][j] *= escalar;
         }
      }
   }
   
   private void multilpicarEscalar(float[][] matriz, float escalar) {
      for (int i = 0; i < matriz.length; i++) {
         for (int j = 0; j < matriz[i].length; j++) {
            matriz[i][j] *= escalar;
         }
      }
   }
   
   private void multilpicarEscalar(double[][] matriz, double escalar) {
      for (int i = 0; i < matriz.length; i++) {
         for (int j = 0; j < matriz[i].length; j++) {
            matriz[i][j] *= escalar;
         }
      }
   }

   //hadamard

   public void hadamard(Object a, Object b, Object r) {
      if (a == null || b == null || r == null) {
         throw new IllegalArgumentException("As matrizes fornecidas não podem ser nulas.");
      }

      if ((a instanceof int[][]) && (b instanceof int[][]) && (r instanceof int[][])) {
         int[][] m1 = (int[][]) a;
         int[][] m2 = (int[][]) b;
         int[][] mr = (int[][]) r;
         hadamard(m1, m2, mr);
         
      } else if ((a instanceof float[][]) && (b instanceof float[][]) && (r instanceof float[][])) {
         float[][] m1 = (float[][]) a;
         float[][] m2 = (float[][]) b;
         float[][] mr = (float[][]) r;
         hadamard(m1, m2, mr);
         
      } else if ((a instanceof double[][]) && (b instanceof double[][]) && (r instanceof double[][])) {
         double[][] m1 = (double[][]) a;
         double[][] m2 = (double[][]) b;
         double[][] mr = (double[][]) r;
         hadamard(m1, m2, mr);
         
      } else {
         throw new IllegalArgumentException("Tipos de matrizes fornecidas não suportados.");
      }   
   }

   private void hadamard(int[][] a, int[][] b, int[][] r) {
      dimensoesIguais(a, b, r);

      for (int i = 0; i < r.length; i++) {
         for (int j = 0; j < r[i].length; j++) {
            r[i][j] = a[i][j] * b[i][j];
         }
      }
   }

   private void hadamard(float[][] a, float[][] b, float[][] r) {
      dimensoesIguais(a, b, r);

      for (int i = 0; i < r.length; i++) {
         for (int j = 0; j < r[i].length; j++) {
            r[i][j] = a[i][j] * b[i][j];
         }
      }
   }

   private void hadamard(double[][] a, double[][] b, double[][] r) {
      dimensoesIguais(a, b, r);

      for (int i = 0; i < r.length; i++) {
         for (int j = 0; j < r[i].length; j++) {
            r[i][j] = a[i][j] * b[i][j];
         }
      }
   }
}

