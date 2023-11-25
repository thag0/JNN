package rna.core;

import java.util.Arrays;

/**
 * Classe que representa uma matriz em forma de array com o objetivo
 * de acelerar as operações dentro da rede neural.
 */
public class Mat implements Cloneable{

   /**
    * Quantidade de linhas da matriz.
    */
   public int lin;

   /**
    * Quantidade de colunas da matriz.
    */
   public int col;

   /**
    * Conjunto de dados.
    */
   private double[] dados;

   /**
    * Inicializar uma nova matriz com os dados desejados.
    * @param lin quantidade de linhas da matriz.
    * @param col quantidade de colunas da matriz.
    * @param dados estrutura de dados contendo os elementos.
    */
   public Mat(int lin, int col, double[] dados){
      if(lin*col != dados.length){
         throw new IllegalArgumentException(
            "A quantidade de linhas e colunas não coincide com o " + 
            "tamanho do conjunto de dados fornecido."
         );
      }
      this.lin = lin;
      this.col = col;
      this.dados = dados;
   }

   /**
    * Inicializar uma nova matriz com seus dados vazios.
    * @param lin quantidade de linhas da matriz.
    * @param col quantidade de colunas da matriz.
    */
   public Mat(int lin, int col){
      this(lin, col, new double[lin*col]);
   }

   /**
    * Retorna o índice correspondente dentro do array de 
    * elementos da matriz.
    * @param lin índice da linha.
    * @param col índice da coluna.
    * @return índice correspondente dentro do array baseado 
    * na linha e coluna fornecidas.
    */
   private int indice(int lin, int col){
      //não sei se deixo essas verificações porque são importantes
      //mas elas também pioram o desempenho

      // if(lin < 0 || lin >= this.lin){
      //    throw new IllegalArgumentException(
      //       "Linha fornecida fora de alcance."
      //    );
      // }
      // if(col < 0 || col >= this.col){
      //    throw new IllegalArgumentException(
      //       "Col fornecida fora de alcance."
      //    );
      // }

      return lin*this.col + col;
   }

   /**
    * Retorna o elemento contido na matriz de acordo com os
    * valores de linha e coluna fornecidos.
    * @param lin índice da linha do elemento.
    * @param col índice da coluna do elemento.
    * @return valor contido de acordo com os índices.
    */
   public double dado(int lin, int col){
      return this.dados[indice(lin, col)];
   }

   /**
    * Coloca o elemento fornecido na matriz de acordo com os
    * valores de linha e coluna fornecidos.
    * @param lin índice da linha do elemento.
    * @param col índice da coluna do elemento.
    * @param valor novo valor que será colocado.
    */
   public void editar(int lin, int col, double valor){
      this.dados[indice(lin, col)] = valor;
   }

   /**
    * Subustitui todo o conteúdo da matriz velo valor fornecido.
    * @param valor novo valor que será colocado.
    */
   public void preencher(double valor){
      for(int i = 0; i < this.dados.length; i++){
         this.dados[i] = valor;
      }
   }

   /**
    * Copia todo o conteúdo da matriz fornecida para a instância 
    * que usar o método.
    * @param m matriz com os dados.
    */
   public void copiar(Mat m){
      this.dados = Arrays.copyOf(this.dados, this.dados.length);
   }
   
   /**
    * Copia todo o conteúdo contido na linha indicada.
    * @param lin índice da linha desejada.
    * @param dados novos dados que serão escritos na linha. 
    */
   public void substituir(int lin, double[] dados){
      int id;
      for(int i = 0; i < this.col; i++){
         id = indice(lin, i);
         this.dados[id] = dados[i];
      }
   }

   /**
    * Adiciona o valor fornecido ao que estiver contido no
    * conteúdo da matriz, de acordo com os índices dados.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    * m[i][j] += d;
    * </pre>
    * @param lin índice da linha.
    * @param col índice da coluna.
    * @param valor dado que será adicionado.
    */
   public void add(int lin, int col, double valor){
      int id = indice(lin, col);
      this.dados[id] += valor;
   }

   /**
    * Subtrai o valor fornecido ao que estiver contido no
    * conteúdo da matriz, de acordo com os índices dados.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    * m[i][j] -= d;
    * </pre>
    * @param lin índice da linha.
    * @param col índice da coluna.
    * @param valor dado que será subtraído.
    */
   public void sub(int lin, int col, double valor){
      int id = indice(lin, col);
      this.dados[id] -= valor;
   }

   /**
    * Multiplica o valor fornecido ao que estiver contido no
    * conteúdo da matriz, de acordo com os índices dados.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    * m[i][j] *= d;
    * </pre>
    * @param lin índice da linha.
    * @param col índice da coluna.
    * @param valor dado que será multiplicado.
    */
  public void mult(int lin, int col, double valor){
      int id = indice(lin, col);
      this.dados[id] *= valor;
   }

   /**
    * Divide o valor fornecido ao que estiver contido no
    * conteúdo da matriz, de acordo com os índices dados.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    * m[i][j] /= d;
    * </pre>
    * @param lin índice da linha.
    * @param col índice da coluna.
    * @param valor dado que será divido.
    */
  public void div(int lin, int col, double valor){
      int id = indice(lin, col);
      this.dados[id] /= valor;
   }

   /**
    * Retorna o conteúdo da linha indicada.
    * @param lin índice da linha.
    * @return array contendo os valores dentro da linha 
    * desejada.
    */
   public double[] linha(int lin){
      int inicio = lin * this.col;
      double[] linha = new double[this.col];
      System.arraycopy(this.dados, inicio, linha, 0, this.col);
      return linha;
   }

   /**
    * Exibe o conteúdo contido na matriz.
    */
   public void print(){
      for(int i = 0; i < this.lin; i++){
         for(int j = 0; j < this.col; j++){
            System.out.print(this.dado(i, j) + " ");
         }
         System.out.println();
      }
   }

   public Mat clone(){
      try{
         Mat clone = (Mat) super.clone();

         clone.lin = this.lin;
         clone.col = this.col;
         clone.dados = Arrays.copyOf(this.dados, this.dados.length);

         return clone;
      }catch(CloneNotSupportedException e){
         throw new RuntimeException(e);
      }
   }
}
