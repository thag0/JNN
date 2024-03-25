package rna.core;

import java.util.function.DoubleUnaryOperator;

/**
 * <h2>
 *    Tensor quadridimensional
 * </h2>
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
 * <h2>
 *    Exemplo de criação:
 * </h2>
 * <pre>
 *Tensor4D tensor = new Tensor4D(1, 1, 2, 2);
 *Tensor4D tensor = new Tensor4D(new int[]{2, 2});
 *Tensor4D tensor = new Tensor4D(2, 2);
 *tensor = [
 *  [[[0.0, 0.0],
 *    [0.0, 0.0]]]
 *]
 * </pre>
 * Algumas operações entre tensores são válidas desde que as dimensões
 * de ambos os tensores sejam iguais.
 * <pre>
 *Tensor4D a = new Tensor4D(1, 1, 2, 2);
 *a.preencer(1);
 *Tensor4D b = new Tensor4D(1, 1, 2, 2);
 *b.preencer(2);
 *a.add(b);//operação acontece dentro do tensor A
 *a = [
 *  [[[3.0, 3.0],
 *    [3.0, 3.0]]]
 *]
 * </pre>
 * @author Thiago Barroso, acadêmico de Engenharia da Computação pela Universidade Federal do Pará, 
 * Campus Tucuruí. Fevereiro/2023.
 */
public class Tensor4D{

   /**
    * Dimensões do tensor (d1, d2, d3, d4).
    */
   private final int[] dimensoes;

   /**
    * Conjunto de elementos do tensor.
    */
   private final double[] dados;

   /**
    * Estético e com finalidade de debug, nome do tensor.
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

      this.dimensoes = new int[4];
      copiarDimensoes(tensor.shape());
      this.dados = new double[tensor.tamanho()];

      System.arraycopy(tensor.dados, 0, this.dados, 0, tamanho());
   }

   /**
    * Inicializa um tensor com quatro dimensões a partir de um tensor quadridimensional
    * primitivo.
    * <p>
    *    O formato do tensor criado será:
    * </p>
    * <pre>
    *    formato = (dim1, dim2, dim3, dim4)
    * </pre>
    * @param tensor tensor desejado.
    */
   public Tensor4D(double[][][][] tensor){
      if(tensor == null){
         throw new IllegalArgumentException(
            "\nO tensor fornecido é nulo."
         );
      }

      this.dimensoes = new int[4];
      copiarDimensoes(
         tensor.length, 
         tensor[0].length, 
         tensor[0][0].length, 
         tensor[0][0][0].length
      );

      this.dados = new double[dim1()*dim2()*dim3()*dim4()];

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

      this.dimensoes = new int[4];
      copiarDimensoes(tensor.length, tensor[0].length, tensor[0][0].length);
      this.dados = new double[dim1()*dim2()*dim3()*dim4()];

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

      this.dimensoes = new int[4];
      copiarDimensoes(matriz.length, matriz[0].length);
      this.dados = new double[dim1()*dim2()*dim3()*dim4()];

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

      this.dimensoes = new int[4];
      copiarDimensoes(array.length);
      this.dados = new double[dim1()*dim2()*dim3()*dim4()];

      System.arraycopy(array, 0, this.dados, 0, array.length);
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

      this.dimensoes = new int[4];
      copiarDimensoes(dim1, dim2, dim3, dim4);
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
    * @param dim array de dimensões contendo os valores em ordem crescente
    * das dimensões do tensor (d1, d2, d3, d4)
    */
   public Tensor4D(int... dim){
      if(dim.length == 0){
         throw new IllegalArgumentException(
            "\nA quantidade de dimensões deve ser maior que zero."
         );
      }
      if(dim.length > 4){
         throw new IllegalArgumentException(
            "\nA quantidade de dimensões deve conter no máximo quatro elementos, " +
            "recebido " + shape().length + "."
         );
      }

      for(int i = 0; i < dim.length; i++){
         if(dim[i] < 1){
            throw new IllegalArgumentException(
               "\nOs valores de dimensões devem ser maiores que zero."
            );
         }
      }

      this.dimensoes = new int[4];
      copiarDimensoes(dim);
      this.dados = new double[dim1()*dim2()*dim3()*dim4()];
   }

   /**
    * Auxliar para cópia de um array de dimensões para as dimensões do tensor.
    * @param dim array de dimensões.
    */
   private void copiarDimensoes(int... dim){
      int n1 = this.dimensoes.length;
      for(int i = 0; i < n1; i++){
         this.dimensoes[i] = 1;
      }
      
      int n2 = dim.length;
      for(int i = 0; i < n2; i++){
         this.dimensoes[n1 - 1 - i] = dim[n2 - 1 - i];
      }
   }

   /**
    * Configura o novo formato para o tensor.
    * <p>
    *    A configuração não altera o conteúdo do tensor, e sim a forma
    *    como os dados são tratados e acessados.
    * </p>
    * Exemplo:
    * <pre>
    *tensor = [
    *  [
    *    [[1, 2],
    *     [3, 4]]
    *  ]  
    *]
    *
    *int[] novoFormato = {1, 1, 1, 4};
    *tensor.reformatar(novoFormato);
    *
    *tensor = [
    *  [
    *    [[1, 2, 3, 4]]
    *  ]  
    *]
    * </pre>
    * @param indices array contendo os novos índices (dim1, dim2, dim3, dim4).
    */
   public void reformatar(int... indices){
      if(indices.length == 0 || indices.length > 4){
         throw new IllegalArgumentException(
            "\nQuantidade de índices deve ser de no máximo 4, recebido " + 
            indices.length + "."
         );
      }

      //novas dimensões
      int[] dims = {1, 1, 1, 1};

      int tam1 = dims.length;
      int tam2 = indices.length;
      for(int i = 0; i < tam2; i++){
         dims[tam1 - 1 - i] = indices[tam2 - 1 - i];
      }

      if((dims[0] < 1) || (dims[1] < 1) || (dims[2] < 1) || (dims[3] < 1)){
         throw new IllegalArgumentException(
            "\nOs novos valores de dimensões devem ser maiores que zero."
         );
      }

      if((dims[0]*dims[1]*dims[2]*dims[3]) != tamanho()){
         throw new IllegalArgumentException(
            "\nA quatidade de elementos com as novas dimensões (" + (dims[0]*dims[1]*dims[2]*dims[3]) + 
            ") deve ser igual a quantidade de elementos do tensor (" + tamanho() + ")."
         );
      }

      copiarDimensoes(dims[0], dims[1], dims[2], dims[3]);
   }

   /**
    * Calcula o índice do elemento dentro do array de elementos do tensor.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param dim3 índice da terceira dimensão.
    * @param dim4 índice da quarta dimensão.
    * @return índice calculado.
    */
   private int indice(int dim1, int dim2, int dim3, int dim4){
      return (dim1 * dim2() * dim3() * dim4()) + 
             (dim2 * dim3() * dim4()) + 
             (dim3 * dim4()) + 
              dim4;
   }

   /**
    * Retorna o elemento do tensor de acordo com os índices fornecidos.
    * <p>
    *    O formato dos índices deve se dar por: {@code (dim1, dim2, dim3, dim4)}
    * </p>
    * @param indices índices desejados para busca.
    * @return valor de acordo com os índices.
    */
   public double get(int... indices){
      if(indices.length != 4){
         throw new IllegalArgumentException(
            "\nA quantidade de índices deve ser igual a 4, recebido " + 
            indices.length + "."
         );
      }

      return dados[indice(indices[0], indices[1], indices[2], indices[3])];
   }

   /**
    * Edita o valor do tensor usando o valor informado.
    * <p>
    *    O formato dos índices deve se dar por: {@code (dim1, dim2, dim3, dim4)}
    * </p>
    * @param indices índices para atribuição.
    * @param valor valor desejado.
    */
   public void set(double valor, int... indices){
      if(indices.length != 4){
         throw new IllegalArgumentException(
            "\nA quantidade de índices deve ser igual a 4, recebido " + 
            indices.length + "."
         );
      }

      dados[indice(indices[0], indices[1], indices[2], indices[3])] = valor;
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
    * Preenche o conteúdo desejado do tensor com um valor constante.
    * @param dim1 índice da primeira dimensão.
    * @param valor valor desejado.
    */
   public void preencher3D(int dim1, double valor){
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }

      int inicio = indice(dim1, 0, 0, 0);
      int fim = inicio + (dim2() * dim3() * dim4());
      for(int i = inicio; i < fim; i++){
         dados[i] = valor;
      }
   }

   /**
    * Preenche o conteúdo desejado do tensor com um valor constante.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param valor valor desejado.
    */
   public void preencher2D(int dim1, int dim2, double valor){
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      int fim = inicio + (dim3() * dim4());
      for(int i = inicio; i < fim; i++){
         dados[i] = valor;
      }
   }

   /**
    * Preenche o conteúdo desejado do tensor com um valor constante.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param dim3 índice da terceira dimensão.
    * @param valor valor desejado.
    */
   public void preencher1D(int dim1, int dim2, int dim3, double valor){
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }
      if(!validarDimensao(dim3, dim3())){
         throw new IllegalArgumentException(
            "\nÍndice da terceira dimensão (" + dim3 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, dim3, 0);
      int fim = inicio + dim4();
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
      int tam = tamanho();
      if(crescente){
         for(int i = 0; i < tam; i++){
            dados[i] = i + 1;
         }

      }else{
         for(int i = 0; i < tam; i++){
            dados[i] = tam - i - 1;
         }
      }
   }

   /**
    * Zera todo o conteúdo o tensor.
    */
   public void zerar(){
      for(int i = 0; i < dados.length; i++){
         dados[i] = 0.0d;
      }
   }

   /**
    * Copia todo o conteúdo do tensor na instância local.
    * @param tensor tensor desejado.
    */
   public void copiar(Tensor4D tensor){
      if(!comparar4D(tensor)){
         throw new IllegalArgumentException(
            "\nDimensões " + shapeStr() + " incompatíveis com as do" + 
            " tensor recebido " + tensor.shapeStr()
         );
      }

      System.arraycopy(tensor.dados, 0, this.dados, 0, tamanho());
   }

   /**
    * Copia o conteúdo do tensor na instância local de acordo a dimensão fornecida.
    * @param tensor tensor desejado.
    * @param dim1 índice da primeira dimensão desejada.
    */
   public void copiar(Tensor4D tensor, int dim1){
      if(!comparar3D(tensor)){
         throw new IllegalArgumentException(
            "\nIncompatibilidade entre as três últimas dimensões do tensor " + shapeStr() +
            " com o tensor fornecido " + tensor.shapeStr()
         );
      }

      int inicio = indice(dim1, 0, 0, 0);
      System.arraycopy(tensor.dados, inicio, this.dados, inicio, (dim2() * dim3() * dim4()));
   }

   /**
    * Copia o conteúdo do tensor na instância local de acordo as dimensões fornecidas.
    * @param tensor tensor desejado.
    * @param dim1 índice da primeira dimensão desejada.
    * @param dim2 índice da segunda dimensão desejada.
    */
   public void copiar(Tensor4D tensor, int dim1, int dim2){
      if(!comparar2D(tensor)){
         throw new IllegalArgumentException(
            "\nIncompatibilidade entre as duas últimas dimensões do tensor " + shapeStr() +
            " com o tensor fornecido " + tensor.shapeStr()
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      System.arraycopy(tensor.dados, inicio, this.dados, inicio, (dim3() * dim4()));
   }

   /**
    * Copia o conteúdo do tensor na instância local de acordo as dimensões fornecidas.
    * @param tensor tensor desejado.
    * @param dim1 índice da primeira dimensão desejada.
    * @param dim2 índice da segunda dimensão desejada.
    * @param dim3 índice da terceira dimensão desejada.
    */
   public void copiar(Tensor4D tensor, int dim1, int dim2, int dim3){
      if(!comparar1D(tensor)){
         throw new IllegalArgumentException(
            "\nIncompatibilidade entre a última dimensão do tensor " + shapeStr() +
            " com o tensor fornecido " + tensor.shapeStr()
         );
      }

      int inicio = indice(dim1, dim2, dim3, 0);
      System.arraycopy(tensor.dados, inicio, this.dados, inicio, dim4());
   }

   /**
    * Copia todo o conteúdo do array na instância local.
    * @param arr array desejado.
    */
   public void copiar(double[][][][] arr){
      if((dim1() != arr.length) ||
         (dim2() != arr[0].length) ||
         (dim3() != arr[0][0].length) ||
         (dim4() != arr[0][0][0].length)
         ){
         throw new IllegalArgumentException(
            "\nDimensões do tensor " + shapeStr() + 
            " incompatíveis com as do array recebido (" 
            + arr.length + ", " + arr[0].length + ", " + arr[0][0].length + ", " + arr[0][0][0].length + ")."
         );
      }

      int cont = 0;
      for(int i = 0; i < dim1(); i++){
         for(int j = 0; j < dim2(); j++){
            for(int k = 0; k < dim3(); k++){
               for(int l = 0; l < dim4(); l++){
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
      if((dim2() != arr.length) ||
         (dim3() != arr[0].length) ||
         (dim4() != arr[0][0].length)
         ){
         throw new IllegalArgumentException(
            "\nDimensões do tensor " + shapeStr() + 
            " incompatíveis com as do array recebido (" 
            + arr.length + ", " + arr[0].length + ", " + arr[0][0].length + ")."
         );
      }

      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 +") inválido."
         );
      }

      int inicio = indice(dim1, 0, 0, 0);
      for(int j = 0; j < dim2(); j++){
         for(int k = 0; k < dim3(); k++){
            for(int l = 0; l < dim4(); l++){
               this.dados[inicio++] = arr[j][k][l];
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
      if((dim3() != arr.length) ||
         (dim4() != arr[0].length)
         ){
         throw new IllegalArgumentException(
            "\nDimensões do tensor " + shapeStr() + 
            " incompatíveis com as do array recebido (" 
            + arr.length + ", " + arr[0].length + ")."
         );
      }

      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 +") inválido."
         );
      }

      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 +") inválido."
         );
      }
      
      for(int i = 0; i < dim3(); i++){
         int inicio = indice(dim1, dim2, i, 0);
         System.arraycopy(arr[i], 0, dados, inicio, dim4());
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
      if(arr.length != dim4()){
         throw new IllegalArgumentException(
            "\nTamanho do array (" + arr.length + 
            ") íncompatível com a capacidade do tensor (" + dim4() + ")."
         );
      }

      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 +") inválido."
         );
      }

      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 +") inválido."
         );
      }

      if(!validarDimensao(dim3, dim3())){
         throw new IllegalArgumentException(
            "\nÍndice da terceira dimensão (" + dim3 +") inválido."
         );
      }

      int inicio = indice(dim1, dim2, dim3, 0);
      System.arraycopy(arr, 0, dados, inicio, dim4());
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
      if(elementos == null){
         throw new IllegalArgumentException(
            "\nArray de elementos não pode ser nulo."
         );
      }

      if(elementos.length != tamanho()){
         throw new IllegalArgumentException(
            "\nTamanho do array fornecido (" + elementos.length + ") inconpatível" +
            "com os elementos do tensor (" + tamanho() + ")."
         );
      }

      System.arraycopy(elementos, 0, dados, 0, tamanho());
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
      if(dim4 && (tensor.dim4() != this.dim4())) return false;
      if(dim3 && (tensor.dim3() != this.dim3())) return false;
      if(dim2 && (tensor.dim2() != this.dim2())) return false;
      if(dim1 && (tensor.dim1() != this.dim1())) return false;

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
    * Aplica a função recebida em todos os elementos do tensor usando
    * como entrada os valores do tensor recebido.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    * tensor.map(x -> Math.random());
    * </pre>
    * Onde {@code x} representa cada elemento dentro do tensor fornecido.
    * @param tensor tensor base.
    * @param funcao função para aplicar no tensor base.
    */
   public void map(Tensor4D tensor, DoubleUnaryOperator funcao){
      if(tensor == null){
         throw new IllegalArgumentException(
            "\nTensor fornecido é nulo."
         );
      }
      if(!comparar4D(tensor)){
         throw new IllegalArgumentException(
            "\nAs dimensões do tensor fornecido " + tensor.shapeStr() +
            " e as da instância local " + shapeStr() + " devem ser iguais."
         );
      }
      if(funcao == null){
         throw new IllegalArgumentException(
            "\nFunção recebida é nula."
         );
      }

      for(int i = 0; i < dados.length; i++){
         dados[i] = funcao.applyAsDouble(tensor.dados[i]);
      }
   }

   /**
    * Aplica a função recebida em todos os elementos do tensor.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    * tensor.map(x -> Math.random());
    * </pre>
    * Onde {@code x} representa cada elemento dentro do tensor.
    * @param fun função desejada.
    */
   public void map(DoubleUnaryOperator fun){
      if(fun == null){
         throw new IllegalArgumentException(
            "\nFunção recebida é nula."
         );
      }

      for(int i = 0; i < dados.length; i++){
         dados[i] = fun.applyAsDouble(dados[i]);
      }
   }

   /**
    * Aplica a função recebida em todos os elementos da primeira dimensão
    * do tensor.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    * tensor.map3D(0, x -> Math.random());
    * </pre>
    * Onde {@code x} representa cada elemento dentro do tensor.
    * @param dim1 índice da primeira dimensão.
    * @param funcao função desejada.
    */
   public void map3D(int dim1, DoubleUnaryOperator funcao){
      if(!validarDimensao(dim1, dim1())){
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
      int fim = inicio + (dim2() * dim3() * dim4());
      for(int i = inicio; i < fim; i++){
         dados[i] = funcao.applyAsDouble(dados[i]);
      }
   }

   /**
    * Aplica a função recebida em todos os elementos da segunda dimensão
    * do tensor.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    * tensor.map2D(0, 0, x -> Math.random());
    * </pre>
    * Onde {@code x} representa cada elemento dentro do tensor.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param funcao função desejada.
    */
   public void map2D(int dim1, int dim2, DoubleUnaryOperator funcao){
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
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
      int fim = inicio + (dim3() * dim4());
      for(int i = inicio; i < fim; i++){
         dados[i] = funcao.applyAsDouble(dados[i]);
      }
   }

   /**
    * Aplica a função recebida em todos os elementos da terceira dimensão
    * do tensor.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    * tensor.map1D(0, 0, 0, x -> Math.random());
    * </pre>
    * Onde {@code x} representa cada elemento dentro do tensor.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @param dim3 índice da terceira dimensão.
    * @param funcao função desejada.
    */
   public void map1D(int dim1, int dim2, int dim3, DoubleUnaryOperator funcao){
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }
      if(!validarDimensao(dim3, dim3())){
         throw new IllegalArgumentException(
            "\nÍndice da terceira dimensão (" + dim3 + ") inválido."
         );
      }
      if(funcao == null){
         throw new IllegalArgumentException(
            "\nFunção recebida é nula."
         );
      }

      int inicio = indice(dim1, dim2, dim3, 0);
      int fim = inicio + dim4();
      for(int i = inicio; i < fim; i++){
         dados[i] = funcao.applyAsDouble(dados[i]);
      }
   }

   /**
    * Retorna a soma de todos os elementos do tensor.
    * @return soma total.
    */
   public double somar(){
      double soma = 0;
      int tam = tamanho();
      for(int i = 0; i < tam; i++){
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
   public double somar3D(int dim1){
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }

      int inicio = indice(dim1, 0, 0, 0);
      int fim = inicio + (dim2() * dim3() * dim4());

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
   public double somar2D(int dim1, int dim2){
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      int fim = inicio + (dim3() * dim4());

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
   public double somar1D(int dim1, int dim2, int dim3){
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }
      if(!validarDimensao(dim3, dim3())){
         throw new IllegalArgumentException(
            "\nÍndice da terceira dimensão (" + dim3 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, dim3, 0);
      int fim = inicio + dim4();

      double soma = 0;
      for(int i = inicio; i < fim; i++){
         soma += dados[i];
      }

      return soma;
   }

   /**
    * Copia o conteúdo de uma linha do tensor e repete ela na quantidade fornecida.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    *tensor = [
    * [
    *  [[1, 2, 3]]  
    * ]  
    *]
    *
    *bloco = bloco2D(0, 0, 0, 3);
    *
    *tensorBloco = [
    * [
    *   [[1, 2, 3],
    *    [1, 2, 3],
    *    [1, 2, 3]]
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
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }
      if(!validarDimensao(dim3, dim3())){
         throw new IllegalArgumentException(
            "\nÍndice da terceira dimensão (" + dim3 + ") inválido."
         );
      }

      Tensor4D bloco = new Tensor4D(1, 1, quantidade, dim4());      
      double[] arr = array1D(dim1, dim2, dim3);

      for(int i = 0; i < quantidade; i++){
         for(int j = 0; j < dim4(); j++){
            bloco.set(arr[j], 0, 0, i, j);
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
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      for(int i = 0; i < dim3(); i++){
         for(int j = 0; j < dim4(); j++){
            set((i == j ? 1 : 0), dim1, dim2, i, j);
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
            "\nDimensões " + shapeStr() + " incompatíveis com as do" + 
            " tensor recebido " + tensor.shapeStr()
         );
      }

      int tam = tamanho();
      for(int i = 0; i < tam; i++){
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
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      int fim = inicio + (dim3() * dim4());
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
            "\nDimensões " + shapeStr() + " incompatíveis com as do" + 
            " tensor recebido " + tensor.shapeStr()
         );
      }

      int tam = tamanho();
      for(int i = 0; i < tam; i++){
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
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      int fim = inicio + (dim3() * dim4());
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
            "\nDimensões " + shapeStr() + " incompatíveis com as do" + 
            " tensor recebido " + tensor.shapeStr()
         );
      }

      int tam = tamanho();
      for(int i = 0; i < tam; i++){
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
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      int fim = inicio + (dim3() * dim4());
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
            "\nDimensões " + shapeStr() + " incompatíveis com as do" + 
            " tensor recebido " + tensor.shapeStr()
         );
      }

      int tam = tamanho();
      for(int i = 0; i < tam; i++){
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
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      int inicio = indice(dim1, dim2, 0, 0);
      int fim = inicio + (dim3() * dim4());
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
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }
      if(!validarDimensao(dim3, dim3())){
         throw new IllegalArgumentException(
            "\nÍndice da terceira dimensão (" + dim3 + ") inválido."
         );
      }

      double[] res = new double[dim4()];

      int inicio = indice(dim1, dim2, dim3, 0);
      System.arraycopy(this.dados, inicio, res, 0, dim4());

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
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      double[][] res = new double[dim3()][dim4()];

      int inicio;
      for(int i = 0; i < dim3(); i++){
         inicio = indice(dim1, dim2, i, 0);
         System.arraycopy(this.dados, inicio, res[i], 0, dim4());
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
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }

      double[][][] res = new double[dim2()][dim3()][dim4()];

      int inicio;
      for(int i = 0; i < dim2(); i++){
         for(int j = 0; j < dim3(); j++){
            inicio = indice(dim1, i, j, 0);
            System.arraycopy(this.dados, inicio, res[i][j], 0, dim4());
         }
      }

      return res;
   }

   /**
    * Retorna um array de quatro dimensões contendo o todo o conteúdo
    * do tensor.
    * @return array contendo os elementos.
    */
   public double[][][][] array4D(){
      double[][][][] res = new double[dim1()][dim2()][dim3()][dim4()];

      for(int i = 0; i < dim1(); i++){
         for(int j = 0; j < dim2(); j++){
            for(int k = 0; k < dim3(); k++){
               for(int l = 0; l < dim4(); l++){
                  res[i][j][k][l] = get(i, j, k, l);
               }
            }
         }
      }

      return res;
   }

   /**
    * Retorna um novo tensor contendo o conteúdo da dimensão especificada.
    * @param dim1 índice da primeira dimensão desejada.
    * @return tensor contendo os sub dados.
    */
   public Tensor4D subTensor3D(int dim1){
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }

      Tensor4D tensor = new Tensor4D(1, dim2(), dim3(), dim4());
   
      int inicio = indice(dim1, 0, 0, 0);      
      System.arraycopy(this.dados, inicio, tensor.dados, 0, (dim2() * dim3() * dim4()));

      return tensor;
   }

   /**
    * Retorna um novo tensor contendo o conteúdo das dimensões especificadas.
    * @param dim1 índice da primeira dimensão desejada.
    * @param dim2 índice da segunda dimensão desejada.
    * @return tensor contendo os sub dados.
    */
   public Tensor4D subTensor2D(int dim1, int dim2){
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }

      Tensor4D tensor = new Tensor4D(1, 1, dim3(), dim4());
   
      int inicio = indice(dim1, dim2, 0, 0);      
      System.arraycopy(this.dados, inicio, tensor.dados, 0, (dim3() * dim4()));

      return tensor;
   }

   /**
    * Retorna um novo tensor contendo o conteúdo das dimensões especificadas.
    * @param dim1 índice da primeira dimensão desejada.
    * @param dim2 índice da segunda dimensão desejada.
    * @param dim3 índice da terceira dimensão desejada.
    * @return tensor contendo os sub dados.
    */
   public Tensor4D subTensor1D(int dim1, int dim2, int dim3){
      if(!validarDimensao(dim1, dim1())){
         throw new IllegalArgumentException(
            "\nÍndice da primeira dimensão (" + dim1 + ") inválido."
         );
      }
      if(!validarDimensao(dim2, dim2())){
         throw new IllegalArgumentException(
            "\nÍndice da segunda dimensão (" + dim2 + ") inválido."
         );
      }
      if(!validarDimensao(dim3, dim3())){
         throw new IllegalArgumentException(
            "\nÍndice da terceira dimensão (" + dim3 + ") inválido."
         );
      }

      Tensor4D tensor = new Tensor4D(1, 1, 1, dim4());
   
      int inicio = indice(dim1, dim2, dim3, 0);      
      System.arraycopy(this.dados, inicio, tensor.dados, 0, dim4());

      return tensor;
   }

   /**
    * Monta as informações de exibição do tensor.
    * @param casas quantidade de casas decimais.
    * @return string formatada.
    */
   private String construirPrint(int casas){
      casas = (casas >= 0) ? casas : 10;
      
      String pad = "   ";

      StringBuilder sb = new StringBuilder();
      sb.append(nome).append(" ").append(shapeStr()).append(" = [\n");
      
      int d1 = dim1();
      for(int i = 0; i < d1; i++){
         sb.append(pad).append("[\n");
         formatarD2(sb, casas, pad, i);
         sb.append(pad).append("]").append((i < d1 - 1) ? ",\n" : "\n");
      }
      sb.append("]\n");
  
      return sb.toString();
  }

   /**
    * Função exclusiva para formatar a segunda dimensão do tensor.
    * @param sb string builder.
    * @param casas quantidade de cadas decimais.
    * @param pad espaçamento.
    * @param i índice do contador da primeira dimensão.
    */
   private void formatarD2(StringBuilder sb, int casas, String pad, int i){
      int d2 = dim2();
      for(int j = 0; j < d2; j++){
         sb.append(pad).append(pad).append("[");
         formatarDimensao3(sb, casas, pad, i, j);
         sb.append("]").append((j < d2 - 1) ? ",\n\n" : "\n");
      }
   }

   /**
    * Função exclusiva para formatar a terceira dimensão do tensor.
    * @param sb string builder.
    * @param casas quantidade de cadas decimais.
    * @param pad espaçamento.
    * @param i índice do contador da primeira dimensão.
    * @param j índice do contador da segunda dimensão.
    */
   private void formatarDimensao3(StringBuilder sb, int casas, String pad, int i, int j){
      int d3 = dim3();
      for(int k = 0; k < d3; k++){
         sb.append((k == 0) ? "[" : (pad + pad + " ["));
         formatarD4(sb, casas, pad, i, j, k);
         sb.append("]").append((k < d3 - 1) ? ",\n" : "");
      }
   }

   /**
    * Função exclusiva para formatar a terceira dimensão do tensor.
    * @param sb string builder.
    * @param casas quantidade de cadas decimais.
    * @param pad espaçamento.
    * @param padNum espaçamento entre valores.
    * @param i índice do contador da primeira dimensão.
    * @param j índice do contador da segunda dimensão.
    * @param k índice do contador da terceira dimensão.
    */
   private void formatarD4(StringBuilder sb, int casas, String pad, int i, int j, int k){
      int d4 = dim4();
      String padNum = "   ";
      
      for(int l = 0; l < d4; l++){
         double valor = dados[indice(i, j, k, l)];
         String formato = String.format("%." + casas + "f", valor);
         sb.append(valor >= 0 ? "+" : "");
         sb.append(formato);
         if (l < d4-1) sb.append(padNum);
      }
   }

   /**
    * Exibe todo o conteúdo do tensor.
    * @param casas quantidade de casas decimais que serão exibidas. Esse
    * valor só {@code será usado caso seja maior que zero}. Por padrão são usadas
    * 8 casas decimais. A quantidade de casas usadas pode afetar a legibilidade
    * do conteúdo.
    */
   public void print(int casas){
      System.out.println(construirPrint(casas));
   }

   /**
    * Exibe todo o conteúdo do tensor.
    */
   public void print(){
      print(8);
   }

   /**
    * Configura o nome do tensor.
    * @param nome novo nome.
    */
   public void nome(String nome){
      if(nome != null){
         nome = nome.trim();
         if(!nome.isEmpty()){
            this.nome = nome;
         }
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
    *    dim = (d1, d2, d3, d4);
    * </pre>
    * @return dimensões do tensor.
    */
   public int[] shape(){
      return new int[]{
         dim1(), dim2(), dim3(), dim4()
      };
   }

   /**
    * Retorna a primeira dimensão do tensor.
    * @return primeira dimensão do tensor.
    */
   public int dim1(){
      return dimensoes[0];
   }

   /**
    * Retorna a segunda dimensão do tensor.
    * @return segunda dimensão do tensor.
    */
   public int dim2(){
      return dimensoes[1];
   }

   /**
    * Retorna a terceira dimensão do tensor.
    * @return terceira dimensão do tensor.
    */
   public int dim3(){
      return dimensoes[2];
   }

   /**
    * Retorna a quarta dimensão do tensor.
    * @return quarta dimensão do tensor.
    */
   public int dim4(){
      return dimensoes[3];
   }

   /**
    * Retorna uma String contendo as dimensões do tensor, seguindo a ordem:
    * <pre>
    *    dim = (d1, d2, d3, d4);
    * </pre>
    * @return dimensões do tensor em formato de String.
    */
   public String shapeStr(){
      return "(" + dim1() + ", " + dim2() + ", " + dim3() + ", " + dim4() + ")";
   }

   /**
    * Verifica se o índice fornecido está dentro dos limites da dimensão
    * do tensor.
    * @param id índice desejado.
    * @param dim dimensão desejada.
    * @return {@code true} caso o índice seja válido, {@code false} caso contrário.
    */
   public boolean validarIndice(int id, int dim){
      return (dim >= 0 && dim < dimensoes.length) && (id >= 0 && id < dimensoes[dim]);
   }

   /**
    * Verifica se o índice fornecido está dentro dos limites da dimensão desejada.
    * @param d dimensão para ser avaliada.
    * @param dim dimensão correspondente do tensor.
    * @return {@code true} caso o índice seja válido, {@code false} caso contrário.
    */
   public boolean validarDimensao(int d, int dim){
      return (d >= 0) && (d <= dim);
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
      Tensor4D clone = new Tensor4D(this);
      return clone;
   }

   @Override
   public String toString(){
      StringBuilder sb = new StringBuilder(construirPrint(8));
      int tamanho = sb.length();

      sb.delete(tamanho-1, tamanho);//remover ultimo "\n"    
      
      sb.append(" <tipo: " + dados.getClass().getComponentType().getSimpleName() + ">");
      sb.append(" <hash: " + Integer.toHexString(hashCode()) + ">");
      sb.append("\n");
      
      return sb.toString();
   }

   @Override
   public boolean equals(Object obj){
      return (obj instanceof Tensor4D) && comparar((Tensor4D) obj);
   }
}
