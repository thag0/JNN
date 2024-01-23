package rna.core;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Impelementações de operações matriciais para uso dentro
 * da biblioteca.
 * <p>
 *    As operações necessitam de objetos do tipo {@code Mat}, que é
 *    um objeto que representa uma matriz bidimensional, mas alocando
 *    os dados num array unidimensional para obter mais desempenho de
 *    execução.
 * </p>
 * A matriz do tipo Mat é importada usando:
 * <pre>
 * import rna.core.Mat;
 * </pre>
 */
public class OpMatriz{

   /**
    * Impelementações de operações matriciais para uso dentro
    * da biblioteca.
    */
   public OpMatriz(){}

   /**
    * Checa se as linhas das matrizes fornecidas são iguais.
    * @param a matriz A.
    * @param b matriz B.
    */
   private void verificarLinhas(Mat a, Mat b){
      if(a.lin() != b.lin()){
         throw new IllegalArgumentException(
            "Linhas de A (" + a.lin() + ") e B (" + b.lin() + ") são diferentes."
         );
      }
   }

   /**
    * Checa se as linhas das matrizes fornecidas são iguais.
    * @param a matriz A.
    * @param b matriz B.
    * @param c matriz C.
    */
   private void verificarLinhas(Mat a, Mat b, Mat c){
      if(a.lin() != b.lin() && a.lin() != c.lin()){
         throw new IllegalArgumentException(
            "Linhas de A (" + a.lin() + 
            "), B (" + b.lin() + 
            ") e C (" + c.lin() + ") são diferentes."
         );
      }
   }

   /**
    * Checa se as colunas das matrizes fornecidas são iguais.
    * @param a matriz A.
    * @param b matriz B.
    */
   private void verificarColunas(Mat a, Mat b){
      if(a.col() != b.col()){
         throw new IllegalArgumentException(
            "Colunas de A (" + a.col() + ") e B (" + b.col() + ") são diferentes."
         );
      }
   }

   /**
    * Checa se as colunas das matrizes fornecidas são iguais.
    * @param a matriz A.
    * @param b matriz B.
    * @param c matriz C.
    */
   private void verificarColunas(Mat a, Mat b, Mat c){
      if(a.col() != b.col() && a.col() != c.col()){
         throw new IllegalArgumentException(
            "Colunas de A (" + a.col() + 
            "), B (" + b.col() + 
            ") e C (" + c.col() + ") são diferentes."
         );
      }
   }

   /**
    * Copia todo o conteúdo a matriz para o destino.
    * @param m matriz com os dados.
    * @param r matriz de destino da cópia.
    */
   public void copiar(Mat m, Mat r){
      if(m.lin() != r.lin()){
         throw new IllegalArgumentException(
            "As linhas de M (" + m.lin() + 
            ") e R (" + r.lin() + 
            ") devem ser iguais"
         );
      }
      if(m.col() != r.col()){
         throw new IllegalArgumentException(
            "As colunas de M (" + m.col() + 
            ") e R (" + r.col() + 
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
      m.preencher(val);
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
      return m.transpor();
   }

   /**
    * Cria uma matriz identidade baseada no tamanho fornecido.
    * @param tamanho tamanho da matriz, valor usado tanto para
    * o número de linhas quanto para o número de colunas.
    * @return matriz identidade.
    */
   public Mat identidade(int tamanho){
      Mat id = new Mat(tamanho, tamanho);
      
      id.forEach((i, j) -> {
         id.editar(i, j, (i == j ? 1 : 0));
      });

      return id;
   }

   /**
    * Encontra na matriz o maior valor dentro da linha fornecida.
    * @param m matriz.
    * @param lin linha desejada.
    */
   public void argmax(Mat m, int lin){
      if(lin < 0 | lin >= m.lin()){
         throw new IllegalArgumentException(
            "Índice de linha (" + lin + ") inválido."
         );
      }

      double[] linha = m.linha(lin);
      int maiorId = 0;
      double maiorValor = linha[0];
      for(int i = 1; i < linha.length; i++){
         if(linha[i] > maiorValor){
            maiorId = i;
            maiorValor = linha[i];
         }
      }

      for(int i = 0; i < linha.length; i++){
         linha[i] = i == maiorId ? 1 : 0;
      }

      m.copiar(lin, linha);
   }

   /**
    * Multiplicação matricial convencional seguindo a expressão:
    * <pre>
    * R = A * B
    * </pre>
    * @param a primeira matriz.
    * @param b segunda matriz.
    * @param r matriz contendo o resultado.
    */
   public void mult(Mat a, Mat b, Mat r){
      if(a.col() != b.lin()){
         throw new IllegalArgumentException("Dimensões de A e B incompatíveis");
      }
      verificarLinhas(a, r);
      verificarColunas(b, r);

      int i, j, k;
      int lin = r.lin(), col = r.col(), acol = a.col();
      double res;

      for(i = 0; i < lin; i++){
         for(j = 0; j < col; j++){
            res = 0;
            for(k = 0; k < acol; k++){
               res += a.elemento(i, k) * b.elemento(k, j);
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
    * @param a primeira matriz.
    * @param b segunda matriz.
    * @param r matriz contendo o resultado.
    */
   public void mult(Mat a, Mat b, Mat r, int nThreads){
      if(a.col() != b.lin()){
         throw new IllegalArgumentException("Dimensões de A e B incompatíveis");
      }
      verificarLinhas(a, r);
      verificarColunas(r, b);

      Thread[] threads = new Thread[nThreads];
      for(int t = 0; t < nThreads; t++){
         final int id = t;

         threads[t] = new Thread(() -> {
            int inicio = id;
            double res;
            int i, j, k;

            for(i = inicio; i < a.lin(); i+=nThreads){
               for(j = 0; j < r.col(); j++){
                  res = 0;
                  for(k = 0; k < a.col(); k++){
                     res += a.elemento(i, k) * b.elemento(k, j);
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
    * @param a primeira matriz.
    * @param b segunda matriz.
    * @param r matriz contendo o resultado da soma.
    */
   public void add(Mat a, Mat b, Mat r){
      r.add(a, b);
   }

   /**
    * Adiciona o conteúdo resultante da soma entre A e B na matriz R de acordo
    * com a expressão:
    * <pre>
    * R = A + B
    * </pre>
    * @param a primeira matriz.
    * @param b segunda matriz.
    * @return matriz contendo o resultado da soma.
    */
   public Mat add(Mat a, Mat b){
      verificarLinhas(a, b);
      verificarColunas(a, b);
      
      Mat r = new Mat(a.lin(), a.col());
      r.add(a, b);

      return r;
   }

   /**
    * Adiciona o conteúdo resultante da subtração entre A e B na 
    * matriz R de acordo com a expressão:
    * <pre>
    * R = A - B
    * </pre>
    * @param a primeira matriz.
    * @param b segunda matriz.
    * @param r matriz contendo o resultado da subtração.
    */
   public void sub(Mat a, Mat b, Mat r){
      r.sub(a, b);
   }

   /**
    * Adiciona o conteúdo resultante da subtração entre A e B na 
    * matriz R de acordo com a expressão:
    * <pre>
    * R = A - B
    * </pre>
    * @param a primeira matriz.
    * @param b segunda matriz.
    * @return r matriz contendo o resultado da subtração.
    */
   public Mat sub(Mat a, Mat b){
      verificarLinhas(a, b);
      verificarColunas(a, b);
      
      Mat r = new Mat(a.lin(), a.col());
      r.sub(a, b);

      return r;
   }

   /**
    * Adiciona o conteúdo resultante do produto elemeto a elemento 
    * entre A e B na matriz R de acordo com a expressão:
    * <pre>
    * R = A ⊙ B
    * </pre>
    * @param a primeira matriz.
    * @param b segunda matriz.
    * @param r matriz contendo o resultado do produto hadamard.
    */
   public void hadamard(Mat a, Mat b, Mat r){
      verificarLinhas(a, b, r);
      verificarColunas(a, b, r);

      r.copiar(a);
      r.mult(b);
   }

   /**
    * Adiciona o conteúdo resultante do produto elemeto a elemento 
    * entre A e B na matriz R de acordo com a expressão:
    * <pre>
    * R = A ⊙ B
    * </pre>
    * @param a primeira matriz.
    * @param b segunda matriz.
    * @return matriz contendo o resultado do produto hadamard.
    */
   public Mat hadamard(Mat a, Mat b){
      verificarLinhas(a, b);
      verificarColunas(a, b);
      
      Mat r = new Mat(a.lin(), a.col());
      r.copiar(a);
      r.mult(b);

      return r;
   }

   /**
    * Adiciona o conteúdo resultante da multiplicação elemento a elemento do conteúdo da matriz
    * A por um valor escalar de acordo com a expressão:
    * <pre>
    * R = A * esc
    * </pre>
    * @param a matriz alvo.
    * @param e escalar usado.
    * @param r matriz que terá o resultado.
    */
   public void multEscalar(Mat a, double e, Mat r){
      verificarLinhas(a, r);
      verificarColunas(a, r);

      r.copiar(a);
      r.map((x) -> x * e);
   }

   /**
    * Adiciona o conteúdo resultante da divisão elemento a elemento do conteúdo da matriz
    * A por um valor escalar de acordo com a expressão:
    * <pre>
    * R = A / esc
    * </pre>
    * @param a matriz alvo.
    * @param e escalar usado.
    * @param r matriz que terá o resultado.
    */
   public void divEscalar(Mat a, double e, Mat r){
      verificarLinhas(a, r);
      verificarColunas(a, r);

      r.copiar(a);
      r.map((x) -> x / e);
   }

   /**
    * Rotaciona o conteúdo da matriz em 180°.
    * @param m matriz.
    */
   public void rotacionar180(Mat m){
      int lin = m.lin();
      int col = m.col();
      Mat rot = new Mat(lin, col);
  
      int i, j;
      for(i = 0; i < lin; i++){
         for(j = 0; j < col; j++){
            rot.editar(i, j, m.elemento(lin - 1 - i, col - 1 - j));
         }
      }
      
      m.copiar(rot);
   } 

   /**
    * Rotaciona o conteúdo da matriz em 180°.
    * @param m matriz.
    * @return nova matriz com o conteúdo rotacionado.
    */
   public Mat rotacionar180R(Mat m){
      int lin = m.lin();
      int col = m.col();
      Mat rot = new Mat(lin, col);
  
      int i, j;
      for(i = 0; i < lin; i++){
         for(j = 0; j < col; j++){
            rot.editar(i, j, m.elemento(lin - 1 - i, col - 1 - j));
         }
      }
      
      return rot;
   } 

   /**
    * Implementação especializada para a propação direta de camadas convolucionais,
    * para que aproveitem um pouco do paralelismo para acelerar o processo de aprendizado.
    * @param entrada conjunto de entradas.
    * @param filtros conjuntos de filtros.
    * @param saida matriz de resultados onde serão armazenados os valores calculados.
    * @param add verificador para o resultado, se {@code verdadeiro} a matriz de resultados
    * não será zerada antes da operação, se {@code falso} a matriz de resultados será
    * zerada antes da operação.
    */
   public void correlacaoCruzada(Mat[] entrada, Mat[][] filtros, Mat[] saida, boolean add){
      if(entrada == null || filtros == null || saida == null){
         throw new IllegalArgumentException("Os valores fornecidos não podem ser nulos.");
      }
      int numFiltros = filtros.length;
      int profEntrada = entrada.length;

      //talvez considerar valores mais otimizados
      int numThreads = (profEntrada > 10) ? Runtime.getRuntime().availableProcessors()/2 : 1;
      ExecutorService executor = Executors.newFixedThreadPool(numThreads);

      for(int i = 0; i < numFiltros; i++){
         final int idFiltro = i;
         executor.submit(() -> {
            for(int j = 0; j < profEntrada; j++){
               correlacaoCruzada(entrada[j], filtros[idFiltro][j], saida[idFiltro], true);
            }
         });
      }

      executor.shutdown();
      try {
         executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
      }catch(InterruptedException e){
         e.printStackTrace();
      }
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
    * Essa operação zera os resultados de r anteriormente.
    * @param a matriz de entrada.
    * @param b filtro ou kernel aplicado na matriz de entrada.
    * @param r resultado da correlação cruzada.
    * @param add verificador para o resultado, se {@code verdadeiro} a matriz de resultados
    * não será zerada antes da operação, se {@code falso} a matriz de resultados será
    * zerada antes da operação.
    */
   public void correlacaoCruzada(Mat a, Mat b, Mat r, boolean add){
      if(r.lin() != (a.lin() - b.lin() + 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as linhas de A (" + a.lin() + 
            "), B (" + b.lin() + 
            ") e R (" + r.lin() + 
            ") incompatíveis."
         );
      }
      if(r.col() != (a.col() - b.col() + 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as colunas de A, B e R incompatíveis."
         );
      }

      if(add == false){
         r.preencher(0);
      }

      int i, j, k, l;
      int rLin = r.lin(), rCol = r.col();
      int bLin = b.lin(), bCol = b.col();
      double res;
      for(i = 0; i < rLin; i++){
         for(j = 0; j < rCol; j++){
            
            res = 0;
            for(k = 0; k < bLin; k++){
               for(l = 0; l < bCol; l++){
                  res += a.elemento(i + k, j + l) * b.elemento(k, l);
               }
            }
            r.add(i, j, res);
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
    * @param r resultado da correlação cruzada.
    * @param add verificador para o resultado, se {@code verdadeiro} a matriz de resultados
    * não será zerada antes da operação, se {@code falso} a matriz de resultados será
    * zerada antes da operação.
    */
   public void correlacaoCruzadaFull(Mat a, Mat b, Mat r, boolean add){
      if(r.lin() != (a.lin() + b.lin() - 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as linhas de A, B e R incompatíveis."
         );
      }
      if(r.col() != (a.col() + b.col() - 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as colunas de A, B e R incompatíveis."
         );
      }

      if(add == false){
         r.preencher(0);
      }
  
      int i, j, k, l, posX, posY;
      double res;
      Mat filtro = rotacionar180R(b);
      for(i = 0; i < r.lin(); i++){
         for(j = 0; j < r.col(); j++){
            res = 0;
            for(k = 0; k < filtro.lin(); k++){
               for (l = 0; l < filtro.col(); l++){
                  posX = i - k;
                  posY = j - l;
  
                  if(posX >= 0 && posX < a.lin() && posY >= 0 && posY < a.col()){
                     res += a.elemento(posX, posY) * filtro.elemento(k, l);
                  }
               }
            }
            r.add(i, j, res);
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
    * @param r resultado da convolução.
    * @param add verificador para o resultado, se {@code verdadeiro} a matriz de resultados
    * não será zerada antes da operação, se {@code falso} a matriz de resultados será
    * zerada antes da operação.
    */
   public void convolucao(Mat a, Mat b, Mat r, boolean add){
      if(r.lin() != (a.lin() - b.lin() + 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as linhas de A, B e R incompatíveis."
         );
      }
      if(r.col() != (a.col() - b.col() + 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as colunas de A, B e R incompatíveis."
         );
      }

      if(add == false){
         r.preencher(0);
      }
      
      int i, j, k, l;
      double res;
      Mat filtro = rotacionar180R(b);
      for(i = 0; i < r.lin(); i++){
         for(j = 0; j < r.col(); j++){
            
            res = 0;
            for(k = 0; k < filtro.lin(); k++){
               for(l = 0; l < filtro.col(); l++){
                  res += a.elemento(i + k, j + l) * filtro.elemento(k, l);
               }
            }
            r.add(i, j, res);
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
    * @param r resultado da convolução.
    * @param add verificador para o resultado, se {@code verdadeiro} a matriz de resultados
    * não será zerada antes da operação, se {@code falso} a matriz de resultados será
    * zerada antes da operação.
    */
   public void convolucaoFull(Mat a, Mat b, Mat r, boolean add){
      if(r.lin() != (a.lin() + b.lin() - 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as linhas de A, B e R incompatíveis."
         );
      }
      if(r.col() != (a.col() + b.col() - 1)){
         throw new IllegalArgumentException(
            "Dimensões entre as colunas de A, B e R incompatíveis."
         );
      }

      if(add == false){
         r.preencher(0);
      }
  
      int i, j, k, l, posX, posY;
      double res;
      for(i = 0; i < r.lin(); i++){
         for(j = 0; j < r.col(); j++){
            res = 0;
            for(k = 0; k < b.lin(); k++){
               for(l = 0; l < b.col(); l++){
                  posX = i - k;
                  posY = j - l;
  
                  if(posX >= 0 && posX < a.lin() && posY >= 0 && posY < a.col()){
                     res += a.elemento(posX, posY) * b.elemento(k, l);
                  }
               }
            }
            r.add(i, j, res);
         }
      }
   }

}
