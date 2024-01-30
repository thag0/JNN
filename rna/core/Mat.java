package rna.core;

import java.util.function.BiConsumer;
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
   private final double[] dados;

   /**
    * Inicializa uma nova matriz com os dados desejados.
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
      if(dados == null){
         throw new IllegalArgumentException(
            "O conjunto de dados não pode ser nulo."
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
    * Inicializa uma nova matriz com seus dados vazios.
    * @param lin quantidade de linhas da matriz.
    * @param col quantidade de colunas da matriz.
    */
   public Mat(int lin, int col){
      this(lin, col, new double[lin*col]);
   }

   /**
    * Inicializa uma nova matriz com seus dados preenchidos de acordo
    * com o valor fornecido.
    * @param lin quantidade de linhas da matriz.
    * @param col quantidade de colunas da matriz.
    * @param valor de preenchimento.
    */
   public Mat(int lin, int col, double valor){
      this(lin, col, new double[lin*col]);
      preencher(valor);
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
      this(1, arr.length, arr);
   }

   /**
    * Inicializa uma nova matriz, copiando o conteúdo da matriz fornecida.
    * @param m matriz base.
    */
   public Mat(Mat m){
      this(m.lin(), m.col(), m.dados.clone());
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
    * Verifica a compatibilidade de dimensões entre a instância local e a 
    * matriz fornecida.
    * @param m matriz que será usada.
    */
   private void verificarDimensoes(Mat m){
      if(this.lin != m.lin || this.col != m.col){
         throw new IllegalArgumentException(
            "Dimensões incompatíveis."
         );   
      }
   }

   /**
    * Verifica a compatibilidade de dimensões entre a instância local e as 
    * matrizes fornecidas.
    * @param a matriz A.
    * @param b matriz B.
    */
   private void verificarDimensoes(Mat a, Mat b){
      if(a.lin != b.lin || a.col != b.col){
         throw new IllegalArgumentException(
            "As dimensões de A (" + a.lin + ", " + a.col + 
            ") e B (" + b.lin + ", " + b.col + ") são incompatíveis."
         );
      }
      
      if(this.lin != a.lin || this.col != a.col){
         throw new IllegalArgumentException(
            "As dimensões de resultado (" + this.lin + ", " + this.col + 
            " incompatível com o esperado (" + a.lin + "," + a.col + ")."
         );
      }
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
      return lin*this.col + col;
   }

   /**
    * Retorna o elemento contido na matriz de acordo com os
    * valores de linha e coluna fornecidos.
    * @param lin índice da linha do elemento.
    * @param col índice da coluna do elemento.
    * @return valor baseado de acordo com os índices.
    */
   public double elemento(int lin, int col){
      return dados[indice(lin, col)];
   }

   /**
    * Coloca o elemento fornecido na matriz de acordo com os
    * valores de linha e coluna fornecidos.
    * @param lin índice da linha do elemento.
    * @param col índice da coluna do elemento.
    * @param valor novo valor que será colocado.
    */
   public void editar(int lin, int col, double valor){
      dados[indice(lin, col)] = valor;
   }

   /**
    * Subustitui todo o conteúdo da matriz velo valor fornecido.
    * @param valor novo valor que será colocado.
    */
   public void preencher(double valor){
      int n = this.tamanho();
      for(int i = 0; i < n; i++){
         dados[i] = valor;
      }
   }

   /**
    * Copia todo o conteúdo da matriz fornecida para a instância 
    * local.
    * <p>
    *    Esse método considera apenas o tamanho total das matrizes,
    *    então é possível copiar matrizes com diferentes formatos 
    *    desde que seus tamanhos finais sejam os mesmos.
    * </p>
    * @param m matriz base.
    */
   public void copiar(Mat m){
      if(this.tamanho() != m.tamanho()){
         throw new IllegalArgumentException(
            "Tamanho de conteúdo de M ("+ m.tamanho() +") " +
            "Não corresponde ao tamanho da matriz (" + this.tamanho() + ")"
         );
      }

      System.arraycopy(m.dados, 0, dados, 0, dados.length);
   }
   
   /**
    * Copia todo o conteúdo da matriz fornecida para a instância 
    * local.
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

      for(int i = 0; i < lin; i++){
         copiar(i, m[i]);
      }
   }

   /**
    * Copia todo o conteúdo contido na linha indicada.
    * @param lin índice da linha desejada.
    * @param dados novos dados que serão escritos na linha. 
    */
   public void copiar(int lin, double[] dados){
      int id = lin*col;
      System.arraycopy(dados, 0, this.dados, id, col);
   }

   /**
    * Copia todo o conteúdo do array fornecido para o array que representa o
    * conjunto de dados da matriz.
    * @param dados
    */
   public void copiar(double[] dados){
      if(this.tamanho() != dados.length){
         throw new IllegalArgumentException(
            "Incompatibilidade de dimensões entre os dados fornecidos (" + dados.length + 
            ") e a instância local (" + this.tamanho() + ")."
         );
      }
      System.arraycopy(dados, 0, this.dados, 0, this.dados.length);
   }

   /**
    * Compara todo o conteúdo da matriz com a instância local.
    * @param m matriz base de comparação.
    * @return true caso os elementos sejam todos iguais, false caso contrário.
    */
   public boolean comparar(Mat m){
      if(this.tamanho() != m.tamanho()){
         throw new IllegalArgumentException(
            "A matriz deve conter a mesma quantidade de elementos (" + m.tamanho() + 
            ") que a instância local (" + this.tamanho() + ")."
         );
      }
      return true;
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
            t.editar(i, j, elemento(j, i));
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
         b.copiar(i, linha(lin));
      }

      return b;
   }

   /**
    * Executa a função fornecida elemento a elemento na matriz.
    * <p>
    *    A função aplicada deve seguir o formato:
    * </p>
    * <pre>
    *(x) -> { 
    *    //"x" é o valor contido na matriz
    *    //correspondente a cada iteração
    *
    *    //processa algo    
    *    return //resultado
    *}
    * </pre>
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    *m = [
    *    1, 2, 3
    *    4, 5, 6
    *    7, 8, 9
    * ]
    *
    *m.aplicarFuncao((x) -> x*2)
    *
    *m = [
    *     2,  4,  6
    *     8, 10, 12
    *    14, 16, 18
    * ]
    * </pre>
    * @param funcao expressão que atuará em cada elemento da matriz.
    */
   public void map(DoubleUnaryOperator funcao){
      if(funcao == null){
         throw new IllegalArgumentException(
            "Função recebida é nula."
         );
      }
      int n = this.tamanho();
      for(int i = 0; i < n; i++){
         dados[i] = funcao.applyAsDouble(dados[i]);
      }
   }

   /**
    * Executa a função fornecida elemento a elemento na matriz e salva o resultado
    * na intância que foi usada.
    * <p>
    *    A função aplicada deve seguir o formato:
    * </p>
    * <pre>
    *(x) -> { 
    *    //"x" é o valor contido na matriz fornecida
    *    //correspondente a cada iteração
    *
    *    //processa algo    
    *    return //resultado
    *}
    * </pre>
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
    *m.aplicarFuncao(a, (x) -> x*2)
    *
    *m = [
    *     2,  4,  6
    *     8, 10, 12
    *    14, 16, 18
    * ]
    * </pre>
    * @param m matriz com os dados de entrada
    * @param funcao expressão que atuará em cada elemento da matriz.
    */
   public void aplicarFuncao(Mat m, DoubleUnaryOperator funcao){
      verificarDimensoes(m);

      int n = this.tamanho();
      for(int i = 0; i < n; i++){
         dados[i] = funcao.applyAsDouble(m.dados[i]);
      }
   }

   /**
    * Percorre cada elemento da matriz e aplica uma operação definida pelo 
    * consumidor, como exemplo:
    * <pre>
    *mat.forEach((i, j) -> {
    *    mat.editar(i, j, i+j);  
    *})
    *
    *mat = [
    *    0, 1, 2 
    *    1, 2, 3 
    *    2, 3, 4 
    *]
    * </pre>
    * @param consumidor operador que vai aplicar a função aos elementos da 
    * matriz, ele recebe dois argumentos (i, j), que se referem
    * aos índices de {@code linha e coluna}, respectivamente.
    */
   public void forEach(BiConsumer<Integer, Integer> consumidor){
      int i, j;
      for(i = 0; i < this.lin; i++){
         for(j = 0; j < this.col; j++){
            consumidor.accept(i, j);
         }
      }
   }

   /**
    * Adiciona o valor fornecido ao que estiver contido no
    * conteúdo da matriz, de acordo com os índices dados.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    * m[i][j] += valor;
    * </pre>
    * @param lin índice da linha.
    * @param col índice da coluna.
    * @param valor dado que será adicionado.
    */
   public void add(int lin, int col, double valor){
      dados[indice(lin, col)] += valor;
   }

   /**
    * Adiciona todo o conteúdo da matriz m localmente.
    * @param m matriz com os dados.
    */
   public void add(Mat m){
      verificarDimensoes(m);

      int n = this.tamanho();
      for(int i = 0; i < n; i++){
         dados[i] += m.dados[i];
      }
   }

   /**
    * Salva o conteúdo de resultante da soma elemento a elemento
    * entre os valores das matrizes A e B de acordo com a expressão.
    * <pre>
    * this = A + B
    * </pre>
    * @param a matriz A.
    * @param b matriz B.
    */
   public void add(Mat a, Mat b){
      verificarDimensoes(a, b);
      
      int n = this.tamanho();
      for(int i = 0; i < n; i++){
         dados[i] = a.dados[i] + b.dados[i];
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
      dados[indice(lin, col)] -= valor;
   }

   /**
    * Subtrai todo o conteúdo da matriz m localmente.
    * @param m matriz com os dados.
    */
   public void sub(Mat m){
      verificarDimensoes(m);

      int n = this.tamanho();
      for(int i = 0; i < n; i++){
         dados[i] -= m.dados[i];
      }
   }

   /**
    * Salva o conteúdo de resultante da subtração elemento a elemento
    * entre os valores das matrizes A e B de acordo com a expressão.
    * <pre>
    * this = A - B
    * </pre>
    * @param a matriz A.
    * @param b matriz B.
    */
   public void sub(Mat a, Mat b){
      verificarDimensoes(a, b);
      
      int n = this.tamanho();
      for(int i = 0; i < n; i++){
         dados[i] = a.dados[i] - b.dados[i];
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
      dados[lin*this.col + col] *= valor;
   }

   /**
    * Multiplica todo o conteúdo da matriz m localmente.
    * @param m matriz com os dados.
    */
   public void mult(Mat m){
      verificarDimensoes(m);

      int n = this.tamanho();
      for(int i = 0; i < n; i++){
         dados[i] *= m.dados[i];
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
      dados[lin*this.col + col] /= valor;
   }

   /**
    * Multiplica todo o conteúdo da matriz pelo valor fornecido.
    * @param esc valor para multiplicação.
    */
   public void multEsc(double esc){
      int n = this.tamanho();
      for(int i = 0; i < n; i++){
         dados[i] *= esc;
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
      System.arraycopy(dados, inicio, linha, 0, this.col);
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
         coluna[i] = elemento(i, col);
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
         m[i] = linha(i);
      }

      return m;
   }

   /**
    * Retorna o conteúdo dos dados da matriz em forma vetorial.
    * @return array contendo os elementos da matriz.
    */
   public double[] paraArray(){
      return dados;
   }

   /**
    * Retorna o tamanho do conjunto de dados suportado pela matriz.
    * @return número total de elementos da matriz.
    */
   public int tamanho(){
      return dados.length;      
   }

   /**
    * Retorna a quantidade de linhas presente na matriz.
    * @return linhas da matriz.
    */
   public int lin(){
      return this.lin;
   }

   /**
    * Retorna a quantidade de colunas presente na matriz.
    * @return colunas da matriz.
    */
   public int col(){
      return this.col;
   }

   /**
    * Exibe o conteúdo contido na matriz.
    * @param nome nome personalizado para exibição.
    * @param casas número de casas decimais de representação.
    */
   public void print(String nome, int casas){
      if(casas <= 0){
         throw new IllegalArgumentException(
            "O número de casas deve ser maior que zero."
         ); 
      }

      StringBuilder sb = new StringBuilder();

      if(nome == null || nome.isBlank() || nome.isEmpty()){
         sb.append(this.getClass().getSimpleName());
      }else{
         sb.append(nome);
      }
      sb.append(" (" + this.lin + ", " + this.col + ") = [\n");

      int compMax = 0;
      String formato = "%." + casas + "f";
      for(int i = 0; i < this.lin; i++){
         for(int j = 0; j < this.col; j++){
            double elemento = elemento(i, j);
            int compAtual = String.format(formato, elemento).length();
            if(compAtual > compMax){
               compMax = compAtual;
            }
         }
      }
  
      for(int i = 0; i < this.lin; i++){
         sb.append(" ");
         for(int j = 0; j < this.col; j++){
            String elemento = String.format("%" + (compMax + 2) + "s", String.format(formato, elemento(i, j)));
            elemento = elemento.replace(",", ".");
            sb.append(elemento);
         }
         sb.append("\n");
      }
      
      sb.append("]");
      System.out.println(sb.toString());
   }

   /**
    * Exibe o conteúdo contido na matriz.
    * @param nome nome personalizado para exibição.
    */
   public void print(String nome){
      print(nome, 16);
   }

   /**
    * Exibe o conteúdo contido na matriz.
    * @param casas número de casas decimais de representação.
    */
   public void print(int casas){
      print(this.getClass().getSimpleName(), casas);
   }

   /**
    * Exibe o conteúdo contido na matriz.
    */
   public void print(){
      print(this.getClass().getSimpleName(), 16);
   }

   /**
    * Cria uma nova matriz contendo os mesmo valores
    * da original.
    * @return nova matriz com os mesmos dados.
    */
   public Mat clone(){
      Mat clone = new Mat(this.lin, this.col);
      System.arraycopy(dados, 0, clone.dados, 0, dados.length);
      return clone;
   }
}
