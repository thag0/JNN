package rna.core;

import java.util.function.DoubleUnaryOperator;

/**
 * Implementação de um array multidimensional (atualmente com no máximo quatro
 * dimentões) com finalidade de simplificar o uso de estrutura de dados dentro
 * da biblioteca.
 * <p>
 *    O tensor possui algumas funções próprias com intuito de aproveitar a velocidade
 *    de processamento usando um único array contendo os dados do dele.
 * </p>
 * <p>
 *    Algumas operações mais elaboradas podem precisar do auxílio da classe {@code OpTensor4D},
 *    que implementa operações entre vários tensores.
 * </p>
 */
public class Tensor4D{

   /**
    * Primeira dimensão do tensor.
    */
   private int d1;
    
   /**
    * Segunda dimensão do tensor.
    */
   private int d2;

   /**
    * Terceira dimensão do tensor.
    */
    private int d3;

   /**
    * Quarta dimensão do tensor
    */
    private int d4;

   /**
    * Conjunto de elementos do tensor.
    */
   final double[] dados;

   /**
    * Estético com finalidade de debug, nome do tensor.
    */
   private String nome = getClass().getSimpleName();

   /**
    * Inicializa um tensor com quatro dimensões a partir de outra instância de 
    * Tensor4D.
    * <p>
    *    O conteúdo do tensor recebido será copiado.
    * </p>
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
    * Inicializa um tensor com quatro dimensões a partir de um tensor quadridimensional
    * primitivo.
    * <p>
    *    O formato do tensor criado será:
    * </p>
    * <pre>
    *    formato = (dim4, dim3, dim2, dim1)
    * </pre>
    * @param tensor tensor desejado.
    */
   public Tensor4D(double[][][][] tensor){
      if(tensor == null){
         throw new IllegalArgumentException(
            "\nO tensor fornecido é nulo."
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
    * <p>
    *    O formato do tensor criado será:
    * </p>
    * <pre>
    *    formato = (1, profundidade, linhas, colunas)
    * </pre>
     * @param tensor tensor desejado.
    */
   public Tensor4D(double[][][] tensor){
      if(tensor == null){
         throw new IllegalArgumentException(
            "\nO tensor fornecido é nulo."
         );
      }

      this.d1 = 1;
      this.d2 = tensor.length;
      this.d3 = tensor[0].length;
      this.d4 = tensor[0][0].length;
      this.dados = new double[d1*d2*d3*d4];

      copiar(tensor, 0);
   }

   /**
    * Inicializa um tensor com quatro dimensões a partir de uma matriz primitiva.
    * <p>
    *    O formato do tensor criado será:
    * </p>
    * <pre>
    *    formato = (1, 1, linhas, colunas)
    * </pre>
    * @param matriz matriz desejado.
    */
   public Tensor4D(double[][] matriz){
      if(matriz == null){
         throw new IllegalArgumentException(
            "\nA matriz fornecida é nula."
         );
      }

      int col = matriz[0].length;
      for(int i = 1; i < matriz.length; i++){
         if(matriz[i].length != col){
            throw new IllegalArgumentException(
               "\nA matriz deve conter a mesma quantidade de linhas para todas as colunas."
            );
         }
      }

      this.d1 = 1;
      this.d2 = 1;
      this.d3 = matriz.length;
      this.d4 = matriz[0].length;
      this.dados = new double[d1*d2*d3*d4];

      copiar(matriz, 0, 0);
   }

   /**
    * Inicializa um tensor com quatro dimensões a partir de um array primitivo.
    * <p>
    *    O formato do tensor criado será:
    * </p>
    * <pre>
    *    formato = (1, 1, 1, tamanhoArray)
    * </pre>
    * @param array array desejado.
    */
   public Tensor4D(double[] array){
      if(array == null){
         throw new IllegalArgumentException(
            "\nO array fornecido é nulo."
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
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param dim3 índice da terceira dimensão.
    * @param dim4 índice da quarta dimensão.
    * @param elementos array de elementos com dados predefinidos.
    */
   public Tensor4D(int dim1, int dim2, int dim3, int dim4, double[] elementos){
      if(dim1 < 1 || dim2 < 1 || dim3 < 1 || dim4 < 1){
         throw new IllegalArgumentException(
            "\nOs valores de dimensões não podem ser menores que 1, recebido: (" +
            dim1 + ", " + dim2 + ", " + dim3 + ", " + dim4 + ")."
         );
      }

      if(elementos == null){
         throw new IllegalArgumentException(
            "\nO array fornecido é nulo."
         );
      }
      
      if((dim1*dim2*dim3*dim4) != elementos.length){
         throw new IllegalArgumentException(
            "\nOs valores de índices não correspondem a quantidade de elementos recebida."
         );
      }

      this.d1 = dim1;
      this.d2 = dim2;
      this.d3 = dim3;
      this.d4 = dim4;

      this.dados = new double[elementos.length];
      System.arraycopy(elementos, 0, dados, 0, dados.length);
   }

   /**
    * Inicializa um tensor com quatro dimensões de acordo com os valores fornecidos.
    * <p>
    *    O conteúdo do tensor estará zerado.
    * </p>
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param dim3 índice da terceira dimensão.
    * @param dim4 índice da quarta dimensão.
    */
   public Tensor4D(int dim1, int dim2, int dim3, int dim4){
      this(dim1, dim2, dim3, dim4, new double[dim1*dim2*dim3*dim4]);
   }

   /**
    * Inicializa um tensor com quatro dimensões de acordo com os valores fornecidos.
    * <p>
    *    O conteúdo do tensor estará zerado.
    * </p>
    * @param dimensoes array de dimensões contendo os valores em ordem crescente
    * das dimensões do tensor (d1, d2, d3, d4)
    */
   public Tensor4D(int[] dimensoes){
      if(dimensoes == null){
         throw new IllegalArgumentException(
            "\tArray de dimensões fornecido é nulo."
         );
      }
      if(dimensoes.length > 4){
         throw new IllegalArgumentException(
            "\nA quantidade de dimensões deve conter no máximo quatro elementos, " +
            "recebido " + dimensoes().length + "."
         );
      }
      if((dimensoes[0] < 1) || (dimensoes[1] < 1) || (dimensoes[2] < 1) || (dimensoes[3] < 1)){
         throw new IllegalArgumentException(
            "\nOs valores de dimensões devem ser maiores que zero."
         );
      }
      
      this.d1 = 1;
      this.d2 = 1;
      this.d3 = 1;
      
      if(dimensoes.length == 1){
         this.d4 = dimensoes[0];
      
      }else if(dimensoes.length == 2){
         this.d3 = dimensoes[0];
         this.d4 = dimensoes[1];
         
      }else if(dimensoes.length == 3){
         this.d2 = dimensoes[0];
         this.d3 = dimensoes[1];
         this.d4 = dimensoes[2];
         
      }else{
         this.d1 = dimensoes[0];
         this.d2 = dimensoes[1];
         this.d3 = dimensoes[2];
         this.d4 = dimensoes[3];
      }

      this.dados = new double[d1*d2*d3*d4];
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
    * Preenche todo o conteúdo do tensor com um valor constante.
    * @param valor valor desejado.
    */
   public void preencher(double valor){
      for(int i = 0; i < dados.length; i++){
         dados[i] = valor;
      }
   }

   /**
    * Preenche o conteúdo do tensor com um valor constante.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param valor valor desejado.
    */
   public void preencher2D(int dim1, int dim2, double valor){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(dim2 < 0 || dim2 >= d2){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      int fim = inicio + (d3*d4);
      for(int i = inicio; i < fim; i++){
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
      if(comparar4D(tensor) == false){
         throw new IllegalArgumentException(
            "\nDimensões " + dimensoesStr() + " incompatíveis com as do" + 
            " tensor recebido " + tensor.dimensoesStr()
         );
      }

      System.arraycopy(tensor.dados, 0, this.dados, 0, this.dados.length);
   }

   /**
    * Copia o conteúdo do tensor na instância local de acordo a dimensão fornecida.
    * @param tensor tensor desejado.
    * @param dim1 índice da primeira dimensão desejada.
    */
   public void copiar(Tensor4D tensor, int dim1){
      if(comparar3D(tensor) == false){
         throw new IllegalArgumentException(
            "\nIncompatibilidade entre as três últimas dimensões do tensor " + dimensoesStr() +
            " com o tensor fornecido " + tensor.dimensoesStr()
         );
      }

      int inicio = indice(dim1, 0, 0, 0);
      System.arraycopy(tensor.dados, inicio, this.dados, inicio, (d2*d3*d4));
   }

   /**
    * Copia o conteúdo do tensor na instância local de acordo as dimensões fornecidas.
    * @param tensor tensor desejado.
    * @param dim1 índice da primeira dimensão desejada.
    * @param dim2 índice da segunda dimensão desejada.
    */
   public void copiar(Tensor4D tensor, int dim1, int dim2){
      if(comparar2D(tensor) == false){
         throw new IllegalArgumentException(
            "\nIncompatibilidade entre as duas últimas dimensões do tensor " + dimensoesStr() +
            " com o tensor fornecido " + tensor.dimensoesStr()
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      System.arraycopy(tensor.dados, inicio, this.dados, inicio, (d3*d4));
   }

   /**
    * Copia o conteúdo do tensor na instância local de acordo as dimensões fornecidas.
    * @param tensor tensor desejado.
    * @param dim1 índice da primeira dimensão desejada.
    * @param dim2 índice da segunda dimensão desejada.
    * @param dim3 índice da terceira dimensão desejada.
    */
   public void copiar(Tensor4D tensor, int dim1, int dim2, int dim3){
      if(comparar1D(tensor) == false){
         throw new IllegalArgumentException(
            "\nIncompatibilidade entre a última dimensão do tensor " + dimensoesStr() +
            " com o tensor fornecido " + tensor.dimensoesStr()
         );
      }

      int inicio = indice(dim1, dim2, dim3, 0);
      System.arraycopy(tensor.dados, inicio, this.dados, inicio, d4);
   }

   /**
    * Copia todo o conteúdo do array na instância local.
    * @param arr array desejado.
    */
   public void copiar(double[][][][] arr){
      if(
         d1 != arr.length ||
         d2 != arr[0].length ||
         d3 != arr[0][0].length ||
         d4 != arr[0][0][0].length
         ){
         throw new IllegalArgumentException(
            "\nDimensões " + dimensoesStr() + " incompatíveis com as do" + 
            " array recebido (" 
            + arr.length + ", " + arr[0].length + ", " + arr[0][0].length + ", " + arr[0][0][0].length + ")."
         );
      }

      int cont = 0;
      for(int i = 0; i < d1; i++){
         for(int j = 0; j < d2; j++){
            for(int k = 0; k < d3; k++){
               for(int l = 0; l < d4; l++){
                  this.dados[cont++] = arr[i][j][k][l];
               }
            }
         }
      }
   }

   /**
    * Copia todo o conteúdo do array na instância local.
    * @param arr array desejado.
    * @param dim1 índice da primeira dimensão.
    */
   public void copiar(double[][][] arr, int dim1){
      if(arr.length != d2){
         throw new IllegalArgumentException(
            "\nTamanho da primeira dimensão array (" + arr.length + 
            ") íncompatível com a capacidade do tensor (" + d2 + ")."
         );
      }
      if(arr[0].length != d3){
         throw new IllegalArgumentException(
            "\nTamanho da segunda dimensão array (" + arr[0].length + 
            ") íncompatível com a capacidade do tensor (" + d3 + ")."
         );
      }
      if(arr[0][0].length != d4){
         throw new IllegalArgumentException(
            "\nTamanho da terceira dimensão array (" + arr[0][0].length + 
            ") íncompatível com a capacidade do tensor (" + d4 + ")."
         );
      }

      int cont = 0;
      for(int j = 0; j < d2; j++){
         for(int k = 0; k < d3; k++){
            for(int l = 0; l < d4; l++){
               this.dados[cont++] = arr[j][k][l];
            }
         }
      }
   }

   /**
    * Copia o conteúdo do array na instância local.
    * @param arr array desejado.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    */
   public void copiar(double[][] arr, int dim1, int dim2){
      if(arr.length != d3){
         throw new IllegalArgumentException(
            "\nTamanho da primeira dimensão array (" + arr.length + 
            ") íncompatível com a capacidade do tensor (" + d3 + ")."
         );
      }
      if(arr[0].length != d4){
         throw new IllegalArgumentException(
            "\nTamanho da segunda dimensão array (" + arr.length + 
            ") íncompatível com a capacidade do tensor (" + d4 + ")."
         );
      }
      
      for(int i = 0; i < d3; i++){
         int inicio = indice(dim1, dim2, i, 0);
         System.arraycopy(arr[i], 0, dados, inicio, d4);
      }
   }

   /**
    * Copia o conteúdo do array na instância local.
    * @param arr array desejado.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param dim3 índice da terceira dimensão.
    */
   public void copiar(double[] arr, int dim1, int dim2, int dim3){
      if(arr.length != d4){
         throw new IllegalArgumentException(
            "\nTamanho do array (" + arr.length + 
            ") íncompatível com a capacidade do tensor (" + d4 + ")."
         );
      }

      int inicio = indice(dim1, dim2, dim3, 0);
      System.arraycopy(arr, 0, dados, inicio, d4);
   }

   /**
    * Copia apenas os dados contidos no array, sem levar em considerasão
    * as dimensões do tensor.
    * <p>
    *    Ainda é necessário que a quantidade de elementos do array seja igual
    *    a quantidade de elementos do tensor.
    * </p>
    * @param elementos array de elementos desejado.
    */
   public void copiarElementos(double[] elementos){
      if(elementos.length != dados.length){
         throw new IllegalArgumentException(
            "\nTamanho do array fornecido (" + elementos.length + ") inconpatível" +
            "com os elementos do tensor (" + dados.length + ")."
         );
      }

      System.arraycopy(elementos, 0, dados, 0, dados.length);
   }

   /**
    * Verifica se as dimensões selecionada são iguais as da instância local.
    * @param tensor tensor base.
    * @param dim1 comparar primeira dimensão.
    * @param dim2 comparar segunda dimensão.
    * @param dim3 comparar terceira dimensão.
    * @param dim4 comparar quarta dimensão.
    * @return resultado da verifcação.
    */
   private boolean compararDimensoes(Tensor4D tensor, boolean dim1, boolean dim2, boolean dim3, boolean dim4){
      if(dim1 && (tensor.d1 != this.d1)) return false;
      if(dim2 && (tensor.d2 != this.d2)) return false;
      if(dim3 && (tensor.d3 != this.d3)) return false;
      if(dim4 && (tensor.d4 != this.d4)) return false;

      return true;
   }

   /**
    * Verifica todas as quatro dimensões do tensor local com os
    * valores de dimensões do tensor recebido.
    * @param tensor tensor alvo.
    * @return resultado da verificação.
    */
   public boolean comparar4D(Tensor4D tensor){
      return compararDimensoes(tensor, true, true, true, true);
   }

   /**
    * Verifica as três últimas dimensões do tensor local com os
    * valores de dimensões do tensor recebido.
    * @param tensor tensor alvo.
    * @return resultado da verificação.
    */
   public boolean comparar3D(Tensor4D tensor){
      return compararDimensoes(tensor, false, true, true, true);
   }

   /**
    * Verifica as duas últimas dimensões do tensor local com os
    * valores de dimensões do tensor recebido.
    * @param tensor tensor alvo.
    * @return resultado da verificação.
    */
   public boolean comparar2D(Tensor4D tensor){
      return compararDimensoes(tensor, false, false, true, true);
   }

   /**
    * Verifica a última dimensõe do tensor local com os
    * valores de dimensões do tensor recebido.
    * @param tensor tensor alvo.
    * @return resultado da verificação.
    */
   public boolean comparar1D(Tensor4D tensor){
      return compararDimensoes(tensor, false, false, false, true);
   }

   /**
    * Compara todo o conteúdo da instância local, isso inclui as {@code dimensões}
    * de cada tensor e seus {@code elementos individuais}.
    * @param tensor tensor base.
    * @return {@code true} caso sejam iguais, {@code false} caso contrário.
    */
   public boolean comparar(Tensor4D tensor){
      if(comparar4D(tensor) == false) return false;

      for(int i = 0; i < dados.length; i++){
         if(dados[i] != tensor.dados[i]) return false;
      }

      return true;
   }

   /**
    * Aplica a função recebida em todos os elementos do tensor.
    * @param funcao função desejada.
    */
   public void map(DoubleUnaryOperator funcao){
      if(funcao == null){
         throw new IllegalArgumentException(
            "\nFunção recebida é nula."
         );
      }

      for(int i = 0; i < dados.length; i++){
         dados[i] = funcao.applyAsDouble(dados[i]);
      }
   }

   /**
    * Aplica a função recebida em todos os elementos da primeira dimensão
    * do tensor.
    * @param dim1 índice da primeira dimensão.
    * @param funcao função desejada.
    */
   public void map3D(int dim1, DoubleUnaryOperator funcao){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(funcao == null){
         throw new IllegalArgumentException(
            "\nFunção recebida é nula."
         );
      }

      int inicio = indice(dim1, 0, 0, 0);
      int fim = inicio + (d2*d3*d4);
      for(int i = inicio; i < fim; i++){
         dados[i] = funcao.applyAsDouble(dados[i]);
      }
   }

   /**
    * Aplica a função recebida em todos os elementos da segunda dimensão
    * do tensor.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param funcao função desejada.
    */
   public void map2D(int dim1, int dim2, DoubleUnaryOperator funcao){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(dim2 < 0 || dim2 >= d2){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }
      if(funcao == null){
         throw new IllegalArgumentException(
            "\nFunção recebida é nula."
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      int fim = inicio + (d3*d4);
      for(int i = inicio; i < fim; i++){
         dados[i] = funcao.applyAsDouble(dados[i]);
      }
   }

   /**
    * Retorna a soma de todos os elementos do tensor.
    * @return soma total.
    */
   public double somarElementos(){
      double soma = 0;
      for(int i = 0; i < dados.length; i++){
         soma += dados[i];
      }

      return soma;
   }

   /**
    * Retorna a soma de todos os elementos das últimas três dimensões
    * do tensor de acordo com o índice especificado.
    * @param dim1 índice da primeira dimensão do tensor.
    * @return soma total.
    */
   public double somarElementos3D(int dim1){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }

      int inicio = indice(dim1, 0, 0, 0);
      int fim = inicio + (d2*d3*d4);

      double soma = 0;
      for(int i = inicio; i < fim; i++){
         soma += dados[i];
      }

      return soma;
   }

   /**
    * Retorna a soma de todos os elementos das últimas duas dimensões
    * do tensor de acordo com os índices especificados.
    * @param dim1 índice da primeira dimensão do tensor.
    * @param dim2 índice da segunda dimensão do tensor.
    * @return soma total.
    */
   public double somarElementos2D(int dim1, int dim2){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(dim2 < 0 || dim2 >= d2){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      int fim = inicio + (d3*d4);

      double soma = 0;
      for(int i = inicio; i < fim; i++){
         soma += dados[i];
      }

      return soma;
   }

   /**
    * Retorna a soma de todos os elementos da última dimensão
    * do tensor de acordo com os índices especificados.
    * @param dim1 índice da primeira dimensão do tensor.
    * @param dim2 índice da segunda dimensão do tensor.
    * @param dim3 índice da terceira dimensão do tensor.
    * @return soma total.
    */
   public double somarElementos1D(int dim1, int dim2, int dim3){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(dim2 < 0 || dim2 >= d2){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }
      if(dim3 < 0 || dim3 >= d3){
         throw new IllegalArgumentException(
            "\nÍndice da terceira dimensão (" + dim3 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, dim3, 0);
      int fim = inicio + d4;

      double soma = 0;
      for(int i = inicio; i < fim; i++){
         soma += dados[i];
      }

      return soma;
   }

   /**
    * Edita o conteúdo do tensor para que o valor fornecido esteja
    * configurado de acordo com os índices fornecidos.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param dim3 índice da terceira dimensão.
    * @param dim4 índice da quarta dimensão.
    * @param valor valor desejado.
    */
   public void editar(int dim1, int dim2, int dim3, int dim4, double valor){
      dados[indice(dim1, dim2, dim3, dim4)] = valor;
   }

   /**
    * Copia o conteúdo de uma linha do tensor e repete ela na quantidade fornecida.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    *tensor = [
    * [
    *  [
    *   1, 2, 3 
    *  ]  
    * ]  
    *]
    *
    *bloco = bloco2D(0, 0, 0, 3);
    *
    *tensorBloco = [
    * [
    *  [
    *   1, 2, 3
    *   1, 2, 3
    *   1, 2, 3
    *  ]  
    * ]  
    *]
    * </pre>
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param dim3 índice da terceira dimensão.
    * @param quantidade quantidade de repetições.
    * @return novo tensor com o resultado.
    */
   public Tensor4D bloco2D(int dim1, int dim2, int dim3, int quantidade){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(dim2 < 0 || dim2 >= d2){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }
      if(dim3 < 0 || dim3 >= d3){
         throw new IllegalArgumentException(
            "\nÍndice da terceira dimensão (" + dim3 + ") inválido."
         );
      }


      Tensor4D bloco = new Tensor4D(1, 1, quantidade, d4);
      
      double[] arr = array1D(dim1, dim2, dim3);

      for(int i = 0; i < quantidade; i++){
         for(int j = 0; j < d4; j++){
            bloco.editar(0, 0, i, j, arr[j]);
         }
      }

      return bloco;
   }

   /**
    * Transoforma o conteúdo das últimas duas dimensões do tensor em uma matriz
    * identidade.
    * <p>
    *    Na matriz identidade todos os valores são zerados e os valores da diagonal
    *    principal são editados para 1.
    * </p>
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    */
   public void identidade2D(int dim1, int dim2){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(dim2 < 0 || dim2 >= d2){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      for(int i = 0; i < d3; i++){
         for(int j = 0; j < d4; j++){
            editar(dim1, dim2, i, j, (i == j ? 1 : 0));
         }
      }
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
      if(comparar4D(tensor) == false){
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
    * Adiciona o valor fornecido ao conteúdo das duas últimas dimensões do tensor.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param valor valor desejado.
    */
   public void add2D(int dim1, int dim2, double valor){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(dim2 < 0 || dim2 >= d2){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      int fim = inicio + (d3*d4);
      for(int i = inicio; i < fim; i++){
         dados[i] += valor;
      }
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
      if(comparar4D(tensor) == false){
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
    * Subtrai o valor fornecido ao conteúdo das duas últimas dimensões do tensor.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param valor valor desejado.
    */
   public void sub2D(int dim1, int dim2, double valor){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(dim2 < 0 || dim2 >= d2){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      int fim = inicio + (d3*d4);
      for(int i = inicio; i < fim; i++){
         dados[i] -= valor;
      }
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
      if(comparar4D(tensor) == false){
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
    * Multiplica o valor fornecido ao conteúdo das duas últimas dimensões do tensor.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param valor valor desejado.
    */
    public void mult2D(int dim1, int dim2, double valor){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(dim2 < 0 || dim2 >= d2){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      int fim = inicio + (d3*d4);
      for(int i = inicio; i < fim; i++){
         dados[i] *= valor;
      }
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
      if(comparar4D(tensor) == false){
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
    * Divide o valor fornecido ao conteúdo das duas últimas dimensões do tensor.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param valor valor desejado.
    */
    public void div2D(int dim1, int dim2, double valor){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(dim2 < 0 || dim2 >= d2){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      int fim = inicio + (d3*d4);
      for(int i = inicio; i < fim; i++){
         dados[i] /= valor;
      }
   }

   /**
    * Retorna o conteúdo do tensor no formato de array
    * @return conteúdo do tensor.
    */
   public double[] paraArray(){
      return this.dados;
   }

   /**
    * Retorna um array de uma dimensão de acordo com os índices
    * especificados.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param dim3 índice da terceira dimensão.
    * @return array contendo os elementos.
    */
   public double[] array1D(int dim1, int dim2, int dim3){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(dim2 < 0 || dim2 >= d2){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }
      if(dim3 < 0 || dim3 >= d3){
         throw new IllegalArgumentException(
            "\nÍndice da terceira dimensão (" + dim3 + ") inválido."
         );
      }

      double[] res = new double[d4];

      int inicio = indice(dim1, dim2, dim3, 0);
      System.arraycopy(this.dados, inicio, res, 0, d4);

      return res;
   }

   /**
    * Retorna um array de duas dimensões de acordo com os índices
    * especificados.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @return array contendo os elementos.
    */
   public double[][] array2D(int dim1, int dim2){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(dim2 < 0 || dim2 >= d2){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      double[][] res = new double[d3][d4];

      int inicio;
      for(int i = 0; i < d3; i++){
         inicio = indice(dim1, dim2, i, 0);
         System.arraycopy(this.dados, inicio, res[i], 0, d4);
      }

      return res;
   }

   /**
    * Retorna um array de três dimensões de acordo com o índice
    * especificado.
    * @param dim1 índice da primeira dimensão.
    * @return array contendo os elementos.
    */
   public double[][][] array3D(int dim1){
      if(dim1 < 0 || dim1 >= d1){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }

      double[][][] res = new double[d2][d3][d4];

      int inicio;
      for(int i = 0; i < d2; i++){
         for(int j = 0; j < d3; j++){
            inicio = indice(dim1, i, j, 0);
            System.arraycopy(this.dados, inicio, res[i][j], 0, d4);
         }
      }

      return res;
   }

   /**
    * Exibe todo o conteúdo do tensor.
    * @param casas quantidade de casas decimais que serão exibidas. Esse
    * valor só {@code será usado caso seja maior que zero}. Por padrão são usadas
    * 15 casas decimais. A quantidade de casas usadas pode afetar a legibilidade
    * do conteúdo.
    */
   public void print(int casas){
      String pad = "   ";
      StringBuilder sb = new StringBuilder();

      casas = (casas > 0) ? casas : 15;

      sb.append(nome + " " + dimensoesStr() + " = [\n");
      for(int i = 0; i < d1; i++){
         sb.append(pad + "[\n");
         for(int j = 0; j < d2; j++){
            sb.append(pad + pad + "[\n");
            for(int k = 0; k < d3; k++){
               sb.append(pad + pad + pad);
               for(int l = 0; l < d4; l++){
                  double valor = dados[indice(i, j, k, l)];
                  String formato = String.format("%." + casas + "f", valor);
                  sb.append(valor >= 0 ? " " : "");//alinhamento de valores negativos e positivos
                  sb.append(formato).append("   ");
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
    * Exibe todo o conteúdo do tensor.
    */
   public void print(){
      print(15);
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

   /**
    * Clona o conteúdo do tensor numa instância separada.
    * @return clone da instância local.
    */
   public Tensor4D clone(){
      try{
         Tensor4D clone = new Tensor4D(this);
         return clone;
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }
}
