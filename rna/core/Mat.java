package rna.core;

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
            "A quantidade de linhas e colunas não coincide com o tamanho do conjunto de dados fornecido."
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
    * Auxiliar
    * @param lin
    * @param col
    * @return
    */
   private int indice(int lin, int col){
      return (lin*this.col + col);
   }

   /**
    * Retorna o elemento contido na matriz de acordo com os
    * valores de linha e coluna fornecidos.
    * @param lin índice da linha do elemento.
    * @param col índice da coluna do elemento.
    * @return valor contido de acordo com os índices.
    */
   public double dado(int lin, int col){
      if(lin < 0 || lin >= this.lin){
         throw new IllegalArgumentException(
            "Linha fornecida fora de alcance."
         );
      }
      if(col < 0 || col >= this.col){
         throw new IllegalArgumentException(
            "Col fornecida fora de alcance."
         );
      }

      return this.dados[indice(lin, col)];
   }

   /**
    * Coloca o elemento fornecido na matriz de acordo com os
    * valores de linha e coluna fornecidos.
    * @param lin índice da linha do elemento.
    * @param col índice da coluna do elemento.
    * @param d novo valor que será colocado.
    */
   public void editar(int lin, int col, double d){
      if(lin < 0 || lin >= this.lin){
         throw new IllegalArgumentException(
            "Linha fornecida fora de alcance."
         );
      }
      if(col < 0 || col >= this.col){
         throw new IllegalArgumentException(
            "Col fornecida fora de alcance."
         );
      }

      this.dados[indice(lin, col)] = d;
   }

   /**
    * Copia todo o conteúdo da matriz fornecida para a instância que
    * usar o método.
    * @param m matriz com os dados.
    */
   public void copiar(Mat m){
      System.arraycopy(m.dados, 0, this.dados, 0, this.dados.length);
   }
   
   /**
    * Copia todo o conteúdo contido na linha indicada.
    * @param lin índice da linha desejada.
    * @param d novos dados que serão escritos na linha. 
    */
   public void substituir(int lin, double[] d){
      for(int i = 0; i < this.col; i++){
         this.dados[indice(lin, i)] = d[i];
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
    * @param d dado que será adicionado.
    */
   public void add(int lin, int col, double d){
      this.editar(lin, col, (this.dado(lin, col) + d));
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
    * @param d dado que será subtraído.
    */
   public void sub(int lin, int col, double d){
      this.editar(lin, col, (this.dado(lin, col) - d));
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
    * @param d dado que será multiplicado.
    */
  public void mult(int lin, int col, double d){
      this.editar(lin, col, (this.dado(lin, col) * d));
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
    * @param d dado que será divido.
    */
  public void div(int lin, int col, double d){
      this.editar(lin, col, this.dado(lin, col) / d);
   }

   /**
    * Retorna o conteúdo da linha indicada.
    * @param lin índice da linha.
    * @return array contendo os valores dentro da linha 
    * desejada.
    */
   public double[] linha(int lin){
      double[] l = new double[this.col];
      for(int i = 0; i < this.col; i++){
         l[i] = this.dado(lin, i);
      }
      return l;
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
         
         clone.dados = new double[this.dados.length];
         System.arraycopy(this.dados, 0, clone.dados, 0, this.dados.length);

         return clone;
      }catch(CloneNotSupportedException e){
         throw new RuntimeException(e);
      }
   }
}
