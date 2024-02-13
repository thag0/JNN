package rna.core;

/**
 * Experimental
 */
public class Tensor4D{

   /**
    * Primeira dimensão do tensor.
    */
    int d1;
    
   /**
    * Segunda dimensão do tensor.
    */
   int d2;

   /**
    * Terceira dimensão do tensor.
    */
   int d3;

   /**
    * Quarta dimensão do tensor
    */
   int d4;

   /**
    * Conjunto de elementos do tensor.
    */
   final double[] dados;

   /**
    * Estético com finalidade de debug, nome do tensor.
    */
   String nome = getClass().getSimpleName();

   /**
    * Inicializa um tensor com quatro dimensões a partir de outra instância de 
    * Tensor4D.
    * @param tensor tensor desejado.
    */
   public Tensor4D(Tensor4D tensor){
      if(tensor == null){
         throw new IllegalArgumentException(
            "O tensor fornecido é nulo."
         );
      }

      this.d1 = tensor.d1;
      this.d2 = tensor.d2;
      this.d3 = tensor.d3;
      this.d4 = tensor.d4;
      this.dados = new double[d1*d2*d3*d4];

      System.arraycopy(tensor.dados, 0, this.dados, 0, this.dados.length);
   }

   /**
    * Inicializa um tensor com quatro dimensões a partir de um tensor primitivo
    * do tipo double[][][][]
    * @param tensor tensor desejado.
    */
   public Tensor4D(double[][][][] tensor){
      if(tensor == null){
         throw new IllegalArgumentException(
            "O tensor fornecido é nulo."
         );
      }

      this.d1 = tensor.length;
      this.d2 = tensor[0].length;
      this.d3 = tensor[0][0].length;
      this.d4 = tensor[0][0][0].length;
      this.dados = new double[d1*d2*d3*d4];

      copiar(tensor);
   }

   /**
    * Inicializa um tensor com quatro dimensões a partir de um tensor tridimensional
    * primitivo.
     * @param tensor tensor desejado.
    */
   public Tensor4D(double[][][] tensor){
      if(tensor == null){
         throw new IllegalArgumentException(
            "O tensor fornecido é nulo."
         );
      }

      this.d1 = 1;
      this.d2 = tensor.length;
      this.d3 = tensor[0].length;
      this.d4 = tensor[0][0].length;
      this.dados = new double[d1*d2*d3*d4];

      int cont = 0;
      for(int i = 0; i < d2; i++){
         for(int j = 0; j < d3; j++){
            for(int k = 0; k < d4; k++){
               dados[cont++] = tensor[i][j][k];
            }
         }
      }
   }

   /**
    * Inicializa um tensor com quatro dimensões a partir de uma matriz primitiva.
    * @param matriz matriz desejado.
    */
   public Tensor4D(double[][] matriz){
      if(matriz == null){
         throw new IllegalArgumentException(
            "A matriz fornecida é nula."
         );
      }

      this.d1 = 1;
      this.d2 = 1;
      this.d3 = matriz.length;
      this.d4 = matriz[0].length;
      this.dados = new double[d1*d2*d3*d4];

      int cont = 0;
      for(int i = 0; i < d3; i++){
         for(int j = 0; j < d4; j++){
            dados[cont++] = matriz[i][j];
         }
      }
   }

   /**
    * Inicializa um tensor com quatro dimensões a partir de um array primitivo.
    * @param array array desejado.
    */
   public Tensor4D(double[] array){
      if(array == null){
         throw new IllegalArgumentException(
            "A matriz fornecida é nula."
         );
      }

      this.d1 = 1;
      this.d2 = 1;
      this.d3 = 1;
      this.d4 = array.length;
      this.dados = new double[d1*d2*d3*d4];

      System.arraycopy(array, 0, this.dados, 0, this.dados.length);
   }

   /**
    * Inicializa um tensor com quatro dimensões de acordo com os valores fornecidos.
    * @param d1 índice da primeira dimensão.
    * @param d2 índice da segunda dimensão.
    * @param d3 índice da terceira dimensão.
    * @param d4 índice da quarta dimensão.
    * @param elementos array de elementos com dados predefinidos.
    */
   public Tensor4D(int d1, int d2, int d3, int d4, double[] elementos){
      if(d1 < 1 || d2 < 1 || d3 < 1 || d4 < 1){
         throw new IllegalArgumentException("Os valores de dimensões não podem ser menores que 1.");
      }

      if(elementos == null){
         throw new IllegalArgumentException("O array fornecido é nulo.");
      }
      
      if((d1*d2*d3*d4) != elementos.length){
         throw new IllegalArgumentException("Os valores de índices não correspondem a quantidade de elementos recebida.");
      }

      this.d1 = d1;
      this.d2 = d2;
      this.d3 = d3;
      this.d4 = d4;

      this.dados = elementos;
   }

   /**
    * Inicializa um tensor com quatro dimensões de acordo com os valores fornecidos.
    * <p>
    *    O conteúdo do tensor estará zerado.
    * </p>
    * @param d1 índice da primeira dimensão.
    * @param d2 índice da segunda dimensão.
    * @param d3 índice da terceira dimensão.
    * @param d4 índice da quarta dimensão.
    */
   public Tensor4D(int d1, int d2, int d3, int d4){
      this(d1, d2, d3, d4, new double[d1 * d2 * d3 * d4]);
   }

   /**
    * Calcula o índice do elemento dentro do array de elementos do tensor.
    * @param i índice da primeira dimensão.
    * @param j índice da segunda dimensão.
    * @param k índice da terceira dimensão.
    * @param l índice da quarta dimensão.
    * @return índice calculado.
    */
   private int indice(int i, int j, int k, int l){
      return i * d2 * d3 * d4 + j * d3 * d4 + k * d4 + l;
   }

   /**
    * Retorna o elemento do tensor de acordo com os índices fornecidos.
    * @param i índice da primeira dimensão.
    * @param j índice da segunda dimensão.
    * @param k índice da terceira dimensão.
    * @param l índice da quarta dimensão.
    * @return valor de acordo com os índices.
    */
   public double elemento(int i, int j, int k, int l){
      return dados[indice(i, j, k, l)];
   }

   /**
    * Preenche o conteúdo do tensor com um valor constante.
    * @param valor valor desejado.
    */
   public void preencher(double valor){
      for(int i = 0; i < dados.length; i++){
         dados[i] = valor;
      }
   }

   /**
    * Preenche o conteúdo do tensor usando um contador iniciado com
    * valor 1 que é alterado a cada elemento.
    * @param crescente contador crescente (1, 2, 3, ...), caso falso o 
    * contador é decrescente (-1, -2, -3, ...)  
    */
   public void preencherContador(boolean crescente){
      if(crescente){
         for(int i = 0; i < dados.length; i++){
            dados[i] = i + 1;
         }
      }else{
         int n = dados.length;
         for(int i = 0; i < n; i++){
            dados[i] = n - i - 1;
         }
      }
   }

   /**
    * Zera o conteúdo o tensor.
    */
   public void zerar(){
      for(int i = 0; i < dados.length; i++){
         dados[i] = 0;
      }
   }

   /**
    * Copia todo o conteúdo do tensor na instância local.
    * @param tensor tensor desejado.
    */
   public void copiar(Tensor4D tensor){
      if(compararDimensoes(tensor) == false){
         throw new IllegalArgumentException(
            "\nDimensões " + dimensoesStr() + " incompatíveis com as do" + 
            " tensor recebido " + tensor.dimensoesStr()
         );
      }

      System.arraycopy(tensor.dados, 0, this.dados, 0, this.dados.length);
   }

   /**
    * Copia todo o conteúdo do tensor na instância local.
    * @param tensor tensor desejado.
    */
   public void copiar(double[][][][] tensor){
      if(
         d1 != tensor.length ||
         d2 != tensor[0].length ||
         d3 != tensor[0][0].length ||
         d4 != tensor[0][0][0].length
         ){
         throw new IllegalArgumentException(
            "\nDimensões " + dimensoesStr() + " incompatíveis com as do" + 
            " tensor recebido (" 
            + tensor.length + ", " + tensor[0].length + ", " + tensor[0][0].length + ", " + tensor[0][0][0].length + ")."
         );
      }

      int cont = 0;
      for(int i = 0; i < d1; i++){
         for(int j = 0; j < d2; j++){
            for(int k = 0; k < d3; k++){
               for(int l = 0; l < d4; l++){
                  this.dados[cont++] = tensor[i][j][k][l];
               }
            }
         }
      }
   }

   /**
    * Verifica todas as quatro dimensões do tensor local com os
    * valores de dimensões do tensor recebido.
    * @param tensor tensor alvo.
    * @return resultado da verificação.
    */
   public boolean compararDimensoes(Tensor4D tensor){
      if(
         (this.d1 != tensor.d1) ||
         (this.d2 != tensor.d2) ||
         (this.d3 != tensor.d3) ||
         (this.d4 != tensor.d4)
      ){
         return false;
      }

      return true;
   }

   /**
    * Compara todo o conteúdo da instância local, isso inclui as {@code dimensões}
    * de cada tensor e seus {@code elementos individuais}.
    * @param tensor tensor base.
    * @return {@code true} caso sejam iguais, {@code false} caso contrário.
    */
   public boolean comparar(Tensor4D tensor){
      if(compararDimensoes(tensor) == false) return false;

      for(int i = 0; i < dados.length; i++){
         if(dados[i] != tensor.dados[i]) return false;
      }

      return true;
   }

   /**
    * Edita o conteúdo do tensor para que o valor fornecido esteja
    * configurado de acordo com os índices fornecidos.
    * @param d1 índice da primeira dimensão.
    * @param d2 índice da segunda dimensão.
    * @param d3 índice da terceira dimensão.
    * @param d4 índice da quarta dimensão.
    * @param valor valor desejado.
    */
   public void editar(int d1, int d2, int d3, int d4, double valor){
      dados[indice(d1, d2, d3, d4)] = valor;
   }

   /**
    * Adiciona todo o conteúdo {@code elemento a elemento} do tensor recebido, seguindo
    * a expressão:
    * <pre>
    *    this += tensor
    * </pre>
    * @param tensor tensor com conteúdo.
    */
   public void add(Tensor4D tensor){
      if(compararDimensoes(tensor) == false){
         throw new IllegalArgumentException(
            "\nDimensões " + dimensoesStr() + " incompatíveis com as do" + 
            " tensor recebido " + tensor.dimensoesStr()
         );
      }

      for(int i = 0; i < dados.length; i++){
         dados[i] += tensor.dados[i];
      }
   }

   /**
    * Adiciona o valor ao conteúdo do tensor de acordo com os índices fornecidos;
    * @param d1 índice da primeira dimensão.
    * @param d2 índice da segunda dimensão.
    * @param d3 índice da terceira dimensão.
    * @param d4 índice da quarta dimensão.
    * @param valor valor desejado.
    */
   public void add(int d1, int d2, int d3, int d4, double valor){
      dados[indice(d1, d2, d3, d4)] += valor;
   }

   /**
    * Subtrai todo o conteúdo {@code elemento a elemento} do tensor recebido, seguindo
    * a expressão:
    * <pre>
    *    this -= tensor
    * </pre>
    * @param tensor tensor com conteúdo.
    */
   public void sub(Tensor4D tensor){
      if(compararDimensoes(tensor) == false){
         throw new IllegalArgumentException(
            "\nDimensões " + dimensoesStr() + " incompatíveis com as do" + 
            " tensor recebido " + tensor.dimensoesStr()
         );
      }

      for(int i = 0; i < dados.length; i++){
         dados[i] -= tensor.dados[i];
      }
   }

   /**
    * Subtrai o valor ao conteúdo do tensor de acordo com os índices fornecidos;
    * @param d1 índice da primeira dimensão.
    * @param d2 índice da segunda dimensão.
    * @param d3 índice da terceira dimensão.
    * @param d4 índice da quarta dimensão.
    * @param valor valor desejado.
    */
   public void sub(int d1, int d2, int d3, int d4, double valor){
      dados[indice(d1, d2, d3, d4)] -= valor;
   }

   /**
    * Multiplica todo o conteúdo {@code elemento a elemento} do tensor recebido, seguindo
    * a expressão:
    * <pre>
    *    this *= tensor
    * </pre>
    * @param tensor tensor com conteúdo.
    */
   public void mult(Tensor4D tensor){
      if(compararDimensoes(tensor) == false){
         throw new IllegalArgumentException(
            "\nDimensões " + dimensoesStr() + " incompatíveis com as do" + 
            " tensor recebido " + tensor.dimensoesStr()
         );
      }

      for(int i = 0; i < dados.length; i++){
         dados[i] *= tensor.dados[i];
      }
   }

   /**
    * Multiplica o valor ao conteúdo do tensor de acordo com os índices fornecidos;
    * @param d1 índice da primeira dimensão.
    * @param d2 índice da segunda dimensão.
    * @param d3 índice da terceira dimensão.
    * @param d4 índice da quarta dimensão.
    * @param valor valor desejado.
    */
   public void mult(int d1, int d2, int d3, int d4, double valor){
      dados[indice(d1, d2, d3, d4)] *= valor;
   }

   /**
    * Divide todo o conteúdo {@code elemento a elemento} do tensor recebido, seguindo
    * a expressão:
    * <pre>
    *    this /= tensor
    * </pre>
    * @param tensor tensor com conteúdo.
    */
   public void div(Tensor4D tensor){
      if(compararDimensoes(tensor) == false){
         throw new IllegalArgumentException(
            "\nDimensões " + dimensoesStr() + " incompatíveis com as do" + 
            " tensor recebido " + tensor.dimensoesStr()
         );
      }

      for(int i = 0; i < dados.length; i++){
         dados[i] /= tensor.dados[i];
      }
   }

   /**
    * Divide o valor ao conteúdo do tensor de acordo com os índices fornecidos, como
    * no exemplo:
    * <pre>
    *    tensor[i][j][k][l] /= valor;
    * </pre>
    * @param d1 índice da primeira dimensão.
    * @param d2 índice da segunda dimensão.
    * @param d3 índice da terceira dimensão.
    * @param d4 índice da quarta dimensão.
    * @param valor valor desejado.
    */
   public void div(int d1, int d2, int d3, int d4, double valor){
      dados[indice(d1, d2, d3, d4)] /= valor;
   }

   /**
    * Retorna o conteúdo do tensor no formato de array
    * @return conteúdo do tensor.
    */
   public double[] paraArray(){
      return this.dados;
   }

   /**
    * Exibe todo o conteúdo do tensor.
    */
   public void print(){
      String pad = "   ";
      StringBuilder sb = new StringBuilder();

      //nem me pergunte
      sb.append(nome + " " + dimensoesStr() + " = [\n");
      for(int i = 0; i < d1; i++){
         sb.append(pad + "[\n");
         for(int j = 0; j < d2; j++){
            sb.append(pad + pad + "[\n");
            for(int k = 0; k < d3; k++){
               sb.append(pad + pad + pad);
               for(int l = 0; l < d4; l++){
                  sb.append(dados[indice(i, j, k, l)] + "  ");
               }
               sb.append("\n");
            }
            sb.append(pad + pad + "]");
            sb.append((j+1 < d2) ? ",\n" : "\n");
         }
         sb.append(pad + "]");
         sb.append((i+1 < d1) ? ",\n" : "\n");
      }

      sb.append("]\n");

      System.out.println(sb.toString());
   }

   /**
    * Configura o nome do tensor.
    * @param nome novo nome.
    */
   public void nome(String nome){
      if(nome != null){
         this.nome = nome;
      }
   }

   /**
    * Retorna o nome do tensor.
    * @return nome do tensor.
    */
   public String nome(){
      return this.nome;
   }

   /**
    * Retorna um array contendo as dimensões do tensor, seguindo a ordem:
    * <pre>
    *    dim = [d1, d2, d3, d4];
    * </pre>
    * @return dimensões do tensor.
    */
   public int[] dimensoes(){
      return new int[]{
         d1, d2 ,d3 ,d4
      };
   }

   /**
    * Retorna a primeira dimensão do tensor.
    * @return primeira dimensão do tensor.
    */
   public int dim1(){
      return d1;
   }

   /**
    * Retorna a segunda dimensão do tensor.
    * @return segunda dimensão do tensor.
    */
   public int dim2(){
      return d2;
   }

   /**
    * Retorna a terceira dimensão do tensor.
    * @return terceira dimensão do tensor.
    */
   public int dim3(){
      return d3;
   }

   /**
    * Retorna a quarta dimensão do tensor.
    * @return quarta dimensão do tensor.
    */
   public int dim4(){
      return d4;
   }

   /**
    * Retorna uma String contendo as dimensões do tensor, seguindo a ordem:
    * <pre>
    *    dim = [d1, d2, d3, d4];
    * </pre>
    * @return dimensões do tensor em formato de String.
    */
   public String dimensoesStr(){
      return "(" + d1 + ", " + d2 + ", " + d3 + ", " + d4 + ")";
   }

   /**
    * Retorna a quantidade total de elementos no tensor.
    * @return número elementos do tensor.
    */
   public int tamanho(){
      return dados.length;
   }
}
