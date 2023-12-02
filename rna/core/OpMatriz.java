package rna.core;

public class OpMatriz{

   /**
    * Impelementações de operações matriciais.
    */
   public OpMatriz(){

   }

   /**
    * Checa se as linhas das matrizes fornecidas são iguais.
    * @param a matriz A.
    * @param b matriz B.
    */
   private void verificarLinhas(Mat a, Mat b){
      if(a.lin != b.lin){
         throw new IllegalArgumentException(
            "Linhas de A (" + a.lin + ") e B (" + b.lin + ") são diferentes."
         );
      }
   }

   /**
    * Checa se as colunas das matrizes fornecidas são iguais.
    * @param a matriz A.
    * @param b matriz B.
    */
   private void verificarColunas(Mat a, Mat b){
      if(a.col != b.col){
         throw new IllegalArgumentException(
            "Colunas de A (" + a.col + ") e B (" + b.col + ") são diferentes."
         );
      }
   }

   /**
    * Copia todo o conteúdo a matriz para o destino.
    * @param m matriz com os dados.
    * @param r matriz de destino da cópia.
    */
   public void copiar(Mat m, Mat r){
      if(m.lin != r.lin){
         throw new IllegalArgumentException(
            "As linhas de M (" + m.lin + 
            ") e R (" + r.lin + 
            ") devem ser iguais"
         );
      }
      if(m.col != r.col){
         throw new IllegalArgumentException(
            "As colunas de M (" + m.col + 
            ") e R (" + r.col + 
            ") devem ser iguais"
         );
      }

      r.copiar(m);
   }

   /**
    * Substitui cada elemento da matriz pelo valor fornecido.
    * @param m matriz.
    * @param val valor desejado para preenchimento.
    */
   public void preencher(Mat m, double val){
      int i, j;
      for(i = 0; i < m.lin; i++){
         for(j = 0; j < m.col; j++){
            m.editar(i, j, val);
         }
      }    
   }

   /**
    * Substitui cada elemento da matriz pelo valor fornecido.
    * @param m matriz.
    * @param val valor desejado para preenchimento.
    */
   public void preencher(double[][] m, double val){
      int i, j;
      for(i = 0; i < m.length; i++){
         for(j = 0; j < m[i].length; j++){
            m[i][j] = val;
         }
      }    
   }

   /**
    * Transpõe a matriz fornecida, invertendo suas linhas e colunas.
    * @param m matriz.
    * @return transposta da matriz alvo.
    */
   public Mat transpor(Mat m){
      Mat t = new Mat(m.col, m.lin);

      int i, j;
      for(i = 0; i < t.lin; i++){
         for(j = 0; j < t.col; j++){
            t.editar(i, j, m.dado(j, i));
         }
      }

      return t;
   }

   /**
    * Cria uma matriz identidade baseada no tamanho fornecido.
    * @param tamanho tamanho da matriz, valor usado tanto para
    * o número de linhas quanto para o número de colunas.
    * @return matriz identidade.
    */
   public Mat identidade(int tamanho){
      Mat id = new Mat(tamanho, tamanho);
      
      for(int i = 0; i < id.lin; i++){
         for(int j = 0; j < id.col; j++){
            id.editar(i, j, (i == j ? 1 : 0));
         }
      }

      return id;
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
   public void mult(Mat a, Mat b, Mat r){
      if(a.col != b.lin){
         throw new IllegalArgumentException("Dimensões de A e B incompatíveis");
      }
      verificarLinhas(a, r);
      verificarColunas(r, b);

      int i, j, k;
      double res;
      for(i = 0; i < r.lin; i++){
         for(j = 0; j < r.col; j++){
            res = 0;
            for(k = 0; k < a.col; k++){
               res += a.dado(i, k) * b.dado(k, j);
            }
            r.editar(i, j, res);    
         }
      }
   }

   /**
    * Multiplicação matricial paralela seguindo a expressão:
    * <pre>
    * R = A * B
    * </pre>
    * @param a primeita matriz.
    * @param b segunda matriz.
    * @param r matriz contendo o resultado.
    */
   public void multT(Mat a, Mat b, Mat r, int nThreads){
      if(a.col != b.lin){
         throw new IllegalArgumentException("Dimensões de A e B incompatíveis");
      }
      verificarLinhas(a, r);
      verificarColunas(r, b);

      int linPorThread = a.lin / nThreads;
      Thread[] threads = new Thread[nThreads];

      for(int t = 0; t < nThreads; t++){
         final int id = t;

         threads[t] = new Thread(() -> {
            int inicio = id * linPorThread;
            int fim = (id == nThreads - 1) ? a.lin : (id + 1) * linPorThread;
            double res;
            int i, j, k;

            for(i = inicio; i < fim; i++){
               for(j = 0; j < r.col; j++){
                  res = 0;
                  for(k = 0; k < a.col; k++){
                     res += a.dado(i, k) * b.dado(k, j);
                  }
                  r.editar(i, j, res);
               }
            }
         });

         threads[t].start();
      }
   
      try{
         for(int i = 0; i < nThreads; i++){
            threads[i].join(0);
         }
      }catch(InterruptedException e){
         e.printStackTrace();
         System.exit(1);
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
   public void add(Mat a, Mat b, Mat r){
      verificarLinhas(a, b);
      verificarColunas(a, b);
      verificarLinhas(a, r);
      verificarColunas(a, r);

      int i, j;
      double d;
      for(i = 0; i < r.lin; i++){
         for(j = 0; j < r.col; j++){
            d = a.dado(i, j) + b.dado(i, j);
            r.editar(i, j, d);
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
    * @return matriz contendo o resultado da soma.
    */
   public Mat addR(Mat a, Mat b){
      verificarLinhas(a, b);
      verificarColunas(a, b);
      Mat r = new Mat(a.lin, a.col);

      int i, j;
      double d;
      for(i = 0; i < r.lin; i++){
         for(j = 0; j < r.col; j++){
            d = a.dado(i, j) + b.dado(i, j);
            r.editar(i, j, d);
         }
      }

      return r;
   }

   /**
    * Adiciona o conteúdo resultante da subtração entre A e B na 
    * matriz R de acordo com a expressão:
    * <pre>
    * R = A - B
    * </pre>
    * @param a primeita matriz.
    * @param b segunda matriz.
    * @param r matriz contendo o resultado da subtração.
    */
   public void sub(Mat a, Mat b, Mat r){
      verificarLinhas(a, b);
      verificarColunas(a, b);
      verificarLinhas(a, r);
      verificarColunas(a, r);

      int i, j;
      double d;
      for(i = 0; i < r.lin; i++){
         for(j = 0; j < r.col; j++){
            d = a.dado(i, j) - b.dado(i, j);
            r.editar(i, j, d);
         }
      }
   }

   /**
    * Adiciona o conteúdo resultante da subtração entre A e B na 
    * matriz R de acordo com a expressão:
    * <pre>
    * R = A - B
    * </pre>
    * @param a primeita matriz.
    * @param b segunda matriz.
    * @return r matriz contendo o resultado da subtração.
    */
   public Mat subR(Mat a, Mat b){
      verificarLinhas(a, b);
      verificarColunas(a, b);
      Mat r = new Mat(a.lin, a.col);

      int i, j;
      double d;
      for(i = 0; i < r.lin; i++){
         for(j = 0; j < r.col; j++){
            d = a.dado(i, j) - b.dado(i, j);
            r.editar(i, j, d);
         }
      }

      return r;
   }

   /**
    * Adiciona o conteúdo resultante do produto elemeto a elemento 
    * entre A e B na matriz R de acordo com a expressão:
    * <pre>
    * R = A ⊙ B
    * </pre>
    * @param a primeita matriz.
    * @param b segunda matriz.
    * @param r matriz contendo o resultado do produto hadamard.
    */
   public void hadamard(Mat a, Mat b, Mat r){
      verificarLinhas(a, b);
      verificarColunas(a, b);
      verificarLinhas(a, r);
      verificarColunas(a, r);

      int i, j;
      double d;
      for(i = 0; i < r.lin; i++){
         for(j = 0; j < r.col; j++){
            d = a.dado(i, j) * b.dado(i, j);
            r.editar(i, j, d);
         }
      }
   }

   /**
    * Adiciona o conteúdo resultante do produto elemeto a elemento 
    * entre A e B na matriz R de acordo com a expressão:
    * <pre>
    * R = A ⊙ B
    * </pre>
    * @param a primeita matriz.
    * @param b segunda matriz.
    * @return matriz contendo o resultado do produto hadamard.
    */
   public Mat hadamardR(Mat a, Mat b){
      verificarLinhas(a, b);
      verificarColunas(a, b);
      Mat r = new Mat(a.lin, a.col);

      int i, j;
      double d;
      for(i = 0; i < r.lin; i++){
         for(j = 0; j < r.col; j++){
            d = a.dado(i, j) * b.dado(i, j);
            r.editar(i, j, d);
         }
      }

      return r;
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
   public void escalar(Mat a, double e, Mat r){
      verificarLinhas(a, r);
      verificarColunas(a, r);

      int i, j;
      for(i = 0; i < r.lin; i++){
         r.copiar(i, a.linha(i));
         for(j = 0; j < r.col; j++){
            r.mult(i, j, e);
         }
      }
   }

   /**
    * 
    * @param m
    */
   public void rotacionar180(Mat m){
      int lin = m.lin;
      int col = m.col;
      Mat rot = new Mat(lin, col);
  
      for(int i = 0; i < lin; i++){
         for(int j = 0; j < col; j++){
            rot.editar(i, j, m.dado(lin - 1 - i, col - 1 - j));
         }
      }
      
      m.copiar(rot);
   } 

   /**
    * 
    * @param m
    * @return
    */
   public Mat rotacionar180R(Mat m){
      int lin = m.lin;
      int col = m.col;
      Mat rot = new Mat(lin, col);
  
      for(int i = 0; i < lin; i++){
         for(int j = 0; j < col; j++){
            rot.editar(i, j, m.dado(lin - 1 - i, col - 1 - j));
         }
      }
      
      return rot;
   } 

   /**
    * Realiza a operação de correlação cruzada (válida) entre a matriz de entrada 
    * e o filtro. Expressada por:
    * <pre>
    *    R = A ⋆ B
    * </pre>
    * O resultado da correlação válida deve ser uma matriz com o seguinte formato:
    * <pre>
    *R.altura = A.altura - B.altura + 1
    *R.largura = A.largura - B.largura + 1
    * </pre>
    * @param a matriz de entrada.
    * @param b filtro ou kernel aplicado na matriz de entrada.
    * @param r resultado da correlação cruzada
    */
   public void correlacaoCruzada(Mat a, Mat b, Mat r){
      if(r.lin != (a.lin - b.lin + 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as linhas de A, B e R incompatíveis."
         );
      }
      if(r.col != (a.col - b.col + 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as colunas de A, B e R incompatíveis."
         );
      }

      double res;
      for(int i = 0; i < r.lin; i++){
         for(int j = 0; j < r.col; j++){
            
            res = 0;
            for(int k = 0; k < b.lin; k++){
               for(int l = 0; l < b.col; l++){
                  res += a.dado(i + k, j + l) * b.dado(k, l);
               }
            }

            r.editar(i, j, res);
         }
      }
   }

   /**
    * Realiza a operação de correlação cruzada (completa) entre a matriz de entrada 
    * e o filtro. Expressada por:
    * <pre>
    *    R = A ⋆ B
    * </pre>
    * O resultado da correlação completa deve ser uma matriz com o seguinte formato:
    * <pre>
    *R.altura = A.altura - B.altura + 1
    *R.largura = A.largura - B.largura + 1
    * </pre>
    * @param a matriz de entrada.
    * @param b filtro ou kernel aplicado na matriz de entrada.
    * @param r resultado da correlação cruzada
    */
   public void correlacaoCruzadaFull(Mat a, Mat b, Mat r){
      if(r.lin != (a.lin + b.lin - 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as linhas de A, B e R incompatíveis."
         );
      }
      if(r.col != (a.col + b.col - 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as colunas de A, B e R incompatíveis."
         );
      }
  
      double res;
      Mat filtro = rotacionar180R(b);
      for(int i = 0; i < r.lin; i++){
         for(int j = 0; j < r.col; j++){
            res = 0;
            for(int k = 0; k < filtro.lin; k++){
               for (int l = 0; l < filtro.col; l++){
                  int posX = i - k;
                  int posY = j - l;
  
                  if(posX >= 0 && posX < a.lin && posY >= 0 && posY < a.col){
                     res += a.dado(posX, posY) * filtro.dado(k, l);
                  }
               }
            }
            r.editar(i, j, res);
         }
      }
   }   

   /**
    * Realiza a operação convolucional (válida) entre a matriz de entrada 
    * e o filtro. Expressada por:
    * <pre>
    *    R = A ∗ B
    * </pre>
    * O resultado da convolução válida deve ser uma matriz com o seguinte formato:
    * <pre>
    *R.altura = A.altura - B.altura + 1
    *R.largura = A.largura - B.largura + 1
    * </pre>
    * @param a matriz de entrada para a operação de convolução.
    * @param b filtro ou kernel aplicado na matriz de entrada.
    * @param r resultado da convolução
    */
   public void convolucao(Mat a, Mat b, Mat r){
      if(r.lin != (a.lin - b.lin + 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as linhas de A, B e R incompatíveis."
         );
      }
      if(r.col != (a.col - b.col + 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as colunas de A, B e R incompatíveis."
         );
      }
      
      double res;
      Mat filtro = rotacionar180R(b);
      for(int i = 0; i < r.lin; i++){
         for(int j = 0; j < r.col; j++){
            
            res = 0;
            for(int k = 0; k < filtro.lin; k++){
               for(int l = 0; l < filtro.col; l++){
                  res += a.dado(i + k, j + l) * filtro.dado(k, l);
               }
            }
            r.editar(i, j, res);
         }
      }
   }

   /**
    * Realiza a operação convolucional (comlpeta) entre a matriz de entrada 
    * e o filtro. Expressada por:
    * <pre>
    *    R = A ∗ B
    * </pre>
    * O resultado da convolução completa deve ser uma matriz com o seguinte formato:
    * <pre>
    *R.altura = A.altura + B.altura - 1
    *R.largura = A.largura + B.largura - 1
    * </pre>
    * @param a matriz de entrada para a operação de convolução.
    * @param b filtro ou kernel aplicado na matriz de entrada.
    * @param r resultado da convolução
    */
   public void convolucaoFull(Mat a, Mat b, Mat r){
      if(r.lin != (a.lin + b.lin - 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as linhas de A, B e R incompatíveis."
         );
      }
      if(r.col != (a.col + b.col - 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as colunas de A, B e R incompatíveis."
         );
      }
  
      double res;
      for(int i = 0; i < r.lin; i++){
         for(int j = 0; j < r.col; j++){
            res = 0;
            for(int k = 0; k < b.lin; k++){
               for (int l = 0; l < b.col; l++){
                  int posX = i - k;
                  int posY = j - l;
  
                  if(posX >= 0 && posX < a.lin && posY >= 0 && posY < a.col){
                     res += a.dado(posX, posY) * b.dado(k, l);
                  }
               }
            }
            r.editar(i, j, res);
         }
      }
   }

}
