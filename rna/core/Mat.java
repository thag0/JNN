package rna.core;

import java.util.function.DoubleUnaryOperator;

/**
 * Classe que representa uma matriz em forma de array com o objetivo
 * de acelerar as operações dentro dos modelos.
 */
public class Mat{

   /**
    * Quantidade de linhas da matriz.
    */
   private int lin;

   /**
    * Quantidade de colunas da matriz.
    */
   private int col;

   /**
    * Conjunto de dados da matriz.
    */
   private double[] dados;

   /**
    * Inicializar uma nova matriz com os dados desejados.
    * @param lin quantidade de linhas da matriz.
    * @param col quantidade de colunas da matriz.
    * @param dados estrutura de dados contendo os elementos.
    */
   public Mat(int lin, int col, double[] dados){
      if(lin < 1 || col < 1){
         throw new IllegalArgumentException(
            "Os valores de linhas e colunas devem ser maiores que zero."
         );
      }
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
    * Inicializa uma nova matriz baseada na matriz fornecida.
    * <p>
    *    A matriz fornecida deve conter o mesmo número de colunas
    *    para todas as suas linhas.
    * </p>
    * A nova matriz respeitará o formato da matriz fornecida.
    * Exemplo:
    * <pre>
    *m = [
    *   1, 2, 3 
    *   4, 5, 6 
    *]
    *
    *mat = [
    *   1, 2, 3 
    *   4, 5, 6 
    *]
    * </pre>
    * @param m matriz base.
    */
   public Mat(double[][] m){
      if(m == null){
         throw new IllegalArgumentException(
            "Matriz fonecida é nula."
         );   
      }

      int n = m[0].length;
      for(int i = 1; i < m.length; i++){
         if(n != m[i].length){
            throw new IllegalArgumentException(
               "As colunas da matriz não podem ter tamanhos diferentes."
            );
         }
      }

      this.lin = m.length;
      this.col = n;
      this.dados = new double[this.lin * this.col];
      
      int indice = 0;
      for(int i = 0; i < this.lin; i++){
         for(int j = 0; j < this.col; j++){
            this.dados[indice++] = m[i][j];
         }
      }
   }

   /**
    * Inicializa uma nova matriz baseada no array fornecido.
    * <p>
    *    A nova matriz terá o número de colunas de acordo com 
    *    o tamanho do array.
    * </p>
    * Exemplo:
    * <pre>
    *arr = [1, 2, 3, 4]
    *
    *mat = [
    *   1, 2, 3, 4 
    *]
    * </pre>
    * @param arr array base.
    */
   public Mat(double[] arr){
      if(arr == null){
         throw new IllegalArgumentException(
            "Array fonecido é nulo."
         );
      }

      this.lin = 1;
      this.col = arr.length;
      this.dados = new double[this.col];
      System.arraycopy(arr, 0, this.dados, 0, this.dados.length);
   }

   /**
    * Configura o novo formato de representação da matriz.
    * <p>
    *    Os dados contidos na matriz são representador com uma estrutura
    *    vetorial, alterar a quantidade de linha ou colunas não interfere
    *    no conteúdo dos dados, apenas na forma como eles são representados.
    * </p>
    * @param lin nova quantidade de linhas.
    * @param col nova quantidade de colunas.
    * @throws IllegalArgumentException se o novo formato for inválido.
    */
   public void configurarFormato(int lin, int col){
      if((lin*col) != this.dados.length){
         throw new IllegalArgumentException(
            "O novo formato deve coinscidir com o tamanho dos dados."
         );
      }

      this.lin = lin;
      this.col = col;
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
    * @return valor baseado de acordo com os índices.
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
    * @param m matriz base.
    */
   public void copiar(Mat m){
      if(m.tamanho() != this.tamanho()){
         throw new IllegalArgumentException(
            "Tamanho de conteúdo de M ("+ m.tamanho() +") " +
            "Não corresponde ao tamanho da matriz (" + this.tamanho() + ")"
         );
      }
      System.arraycopy(m.dados, 0, this.dados, 0, this.dados.length);
   }
   
   /**
    * Copia todo o conteúdo da matriz fornecida para a instância 
    * que usar o método.
    * @param m matriz base.
    */
   public void copiar(double[][] m){
      if(m == null){
         throw new IllegalArgumentException(
            "Matriz fornecida é nula."
         );
      }
      if(this.lin != m.length || this.col != m[0].length){
         throw new IllegalArgumentException(
            "Dimensões incompatíveis."
         );
      }
      for(int i = 0; i < this.lin; i++){
         for(int j = 0; j < this.col; j++){
            this.dados[indice(i, j)] = m[i][j];
         }
      }
   }

   /**
    * Copia todo o conteúdo contido na linha indicada.
    * @param lin índice da linha desejada.
    * @param dados novos dados que serão escritos na linha. 
    */
   public void copiar(int lin, double[] dados){
      int id = lin*this.col;
      System.arraycopy(dados, 0, this.dados, id, this.col);
   }

   /**
    * Copia todo o conteúdo do array fornecido para o array que representa o
    * conjunto de dados da matriz.
    * @param dados
    */
   public void copiar(double[] dados){
      System.arraycopy(dados, 0, this.dados, 0, this.dados.length);
   }

   /**
    * Transpõe o conteúdo da matriz, invertendo suas linhas e colunas.
    * @return matriz transposta.
    */
   public Mat transpor(){
      Mat t = new Mat(this.col, this.lin);

      int i, j;
      for(i = 0; i < t.lin; i++){
         for(j = 0; j < t.col; j++){
            t.editar(i, j, this.dado(j, i));
         }
      }

      return t;
   }

   /**
    * Repete o conteúdo da linha indicada de acordo com o valor
    * de repetições fornecido.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    *mat = [
    *    1, 2, 3
    *    4, 5, 6  
    * ]
    *
    *b = mat.bloco(0, 3)
    *
    *b = [
    *    1, 2, 3
    *    1, 2, 3
    *    1, 2, 3
    * ]
    * </pre>
    * @param lin índice da linha desejada.
    * @param n quantidade de vezes que a linha desejada será repetido.
    * @return uma nova matriz contendo os valores do bloco.
    */
   public Mat bloco(int lin, int n){
      if(lin < 0 || lin >= this.lin){
         throw new IllegalArgumentException(
            "Índice (" + lin + ") inválido ou fora de alcance."
         );
      }
      if(n < 1){
         throw new IllegalArgumentException(
            "Número de repetições (" + lin + ") deve ser maior que zero."
         );
      }

      Mat b = new Mat(n, this.col);
      for(int i = 0; i < b.lin; i++){
         b.copiar(i, this.linha(lin));
      }

      return b;
   }

   /**
    * Executa a função fornecida elemento a elemento na matriz.
    * <p>
    *    Exemplo
    * </p>
    * <pre>
    *m = [
    *    1, 2, 3
    *    4, 5, 6
    *    7, 8, 9
    * ]
    *
    *m.aplicarFuncao((x) -> {x*2})
    *
    *m = [
    *     2,  4,  6
    *     8, 10, 12
    *    14, 16, 18
    * ]
    * </pre>
    * @param f expressão lambda que atuará em cada elemento da matriz.
    */
   public void aplicarFuncao(DoubleUnaryOperator f){
      for(int i = 0; i < this.dados.length; i++){
         this.dados[i] = f.applyAsDouble(this.dados[i]);
      }
   }

   /**
    * Executa a função fornecida elemento a elemento na matriz e salva o resultado
    * na intância que foi usada.
    * <p>
    *    Exemplo
    * </p>
    * <pre>
    *a = [
    *    1, 2, 3
    *    4, 5, 6
    *    7, 8, 9
    * ]
    *
    *m.aplicarFuncao(a, (x) -> {x*2})
    *
    *m = [
    *     2,  4,  6
    *     8, 10, 12
    *    14, 16, 18
    * ]
    * </pre>
    * @param m matriz com os dados de entrada
    * @param f expressão lambda que atuará em cada elemento da matriz.
    */
   public void aplicarFuncao(Mat m, DoubleUnaryOperator f){
      if(this.tamanho() != m.tamanho()){
         throw new IllegalArgumentException(
            "A matriz fornecida deve possuir o mesmo tamanho."
         );
      }

      for(int i = 0; i < this.dados.length; i++){
         this.dados[i] = f.applyAsDouble(m.dados[i]);
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
      this.dados[indice(lin, col)] += valor;
   }

   /**
    * Adiciona todo o conteúdo da matriz m localmente.
    * @param m matriz com os dados.
    */
   public void add(Mat m){
      if(this.tamanho() != m.tamanho()){
         throw new IllegalArgumentException(
            "A matriz fornecida deve conter o mesmo número de elementos."
         );
      }

      for(int i = 0; i < this.dados.length; i++){
         this.dados[i] += m.dados[i];
      }
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
      this.dados[indice(lin, col)] -= valor;
   }

   /**
    * Subtrai todo o conteúdo da matriz m localmente.
    * @param m matriz com os dados.
    */
   public void sub(Mat m){
      if(this.tamanho() != m.tamanho()){
         throw new IllegalArgumentException(
            "A matriz fornecida deve conter o mesmo número de elementos."
         );
      }
      for(int i = 0; i < this.dados.length; i++){
         this.dados[i] -= m.dados[i];
      }
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
      this.dados[indice(lin, col)] *= valor;
   }

   /**
    * Multiplica todo o conteúdo da matriz m localmente.
    * @param m matriz com os dados.
    */
   public void mult(Mat m){
      if(this.tamanho() != m.tamanho()){
         throw new IllegalArgumentException(
            "A matriz fornecida deve conter o mesmo número de elementos."
         );
      }
      for(int i = 0; i < this.dados.length; i++){
         this.dados[i] *= m.dados[i];
      }
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
      this.dados[indice(lin, col)] /= valor;
   }

   /**
    * Multiplica todo o conteúdo da matriz pelo valor fornecido.
    * @param esc valor para multiplicação.
    */
   public void escalar(double esc){
     for(int i = 0; i < this.dados.length; i++){
         this.dados[i] *= esc;
      }
   }

   /**
    * Retorna o conteúdo da linha indicada.
    * @param lin índice da linha.
    * @return array contendo os valores dentro da linha 
    * desejada.
    */
   public double[] linha(int lin){
      if(lin < 0 || lin >= this.lin){
         throw new IllegalArgumentException(
            "Índice de linha (" + lin + ") inválido."
         );
      }

      double[] linha = new double[this.col];
      int inicio = lin * this.col;
      System.arraycopy(this.dados, inicio, linha, 0, this.col);
      return linha;
   }

   /**
    * Retorna o conteúdo da coluna indicada.
    * @param col índice da coluna.
    * @return array contendo os valores dentro da linha 
    * desejada.
    */
   public double[] coluna(int col){
      if(col < 0 || col >= this.col){
         throw new IllegalArgumentException(
            "Índice de coluna (" + col + ") inválido."
         );
      }

      double[] coluna = new double[this.lin];
      for(int i = 0; i < this.lin; i++){
         coluna[i] = this.dado(i, col);
      }

      return coluna;
   }

   /**
    * Converte os dados da matriz num array bidimensional.
    * @return array bidimensional do tipo double.
    */
   public double[][] paraDouble(){
      double[][] m = new double[this.lin][this.col];

      for(int i = 0; i < this.lin; i++){
         m[i] = this.linha(i);
      }

      return m;
   }

   /**
    * Retorna o conteúdo dos dados da matriz em forma vetorial.
    * @return array contendo os elementos da matriz.
    */
   public double[] paraArray(){
      return this.dados;
   }

   /**
    * Retorna o tamanho do conjunto de dados suportado pela matriz.
    * @return número de elementos da matriz.
    */
   public int tamanho(){
      return this.dados.length;      
   }

   /**
    * Retorna a quantidade de linhas presente na matriz.
    * @return linhas da matriz;
    */
   public int lin(){
      return this.lin;
   }

   /**
    * Retorna a quantidade de colunas presente na matriz.
    * @return colunas da matriz;
    */
   public int col(){
      return this.col;
   }

   /**
    * Exibe o conteúdo contido na matriz.
    */
   public void print(String nome){
      if(nome == null || nome.isBlank() || nome.isEmpty()){
          System.out.print(this.getClass().getSimpleName());
      }else{
         System.out.print(nome);
      }
      System.out.println(" (" + this.lin + ", " + this.col + ") = [");

      int compMax = 0;
      for(int i = 0; i < this.lin; i++){
         for(int j = 0; j < this.col; j++){
            int compAtual = String.valueOf(this.dado(i, j)).length();
            if(compAtual > compMax){
               compMax = compAtual;
            }
         }
      }
  
      for(int i = 0; i < this.lin; i++){
         System.out.print(" ");
         for(int j = 0; j < this.col; j++){
            String element = String.format("%" + (compMax + 2) + "s", this.dado(i, j));
            System.out.print(element);
         }
         System.out.println();
      }
  
      System.out.println("]");
   }

   /**
    * Exibe o conteúdo contido na matriz.
    */
   public void print(){
      this.print(this.getClass().getSimpleName());
   }

   /**
    * Cria uma nova matriz contendo os mesmo valores
    * da original.
    * @return nova matriz com os mesmos dados.
    */
   public Mat clone(){
      Mat clone = new Mat(this.lin, this.col);
      System.arraycopy(this.dados, 0, clone.dados, 0, this.dados.length);
      return clone;
   }
}
