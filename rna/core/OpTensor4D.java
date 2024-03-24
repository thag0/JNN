package rna.core;

/**
 * Auxiliar em operação para tensores 4D.
 */
public class OpTensor4D{

   /**
    * Auxiliar na opeação com arrays.
    */
   OpArray oparr = new OpArray();
   
   /**
    * Auxiliar em operação para tensores 4D.
    */
   public OpTensor4D(){}

   /**
    * Verifica se os tensores possuem a mesma altura (dim3) e largura (dim4).
    * @param a tensor A.
    * @param b tensor B.
    * @return resultado da verificação.
    */
   public boolean compararAlturaLargura(Tensor4D a, Tensor4D b){
      return (a.dim3() == b.dim3()) && (a.dim4() == b.dim4());
   }

   /**
    * Copia o conteúdo das duas últimas dimensões do tensor para o destino.
    * @param tensor tensor base.
    * @param destino tensor de destino da cópia.
    * @param dimA índies das duas primeiras dimensões do tensor base (dim1, dim2)
    * @param dimA índies das duas primeiras dimensões do tensor de destino (dim1, dim2)
    */
   public void copiarMatriz(Tensor4D tensor, Tensor4D destino, int[] dimA, int[] dimB){
      if(tensor.comparar2D(destino) == false){
         throw new IllegalArgumentException(
            "\nAs duas últimas dimensões do tensor recebido " + tensor.shapeStr() +
            " e de destino " + destino.shapeStr() + " devem ser iguais."
         );
      }

      if((dimA[0] < 0 || dimA[0] >= tensor.dim1()) || (dimA[1] < 0 || dimA[1] >= tensor.dim2())){
         throw new IllegalArgumentException(
            "\nÍndices do tensor base (" + dimA[0] + ", " + dimA[1] + ") " +
            "inválidos para o tensor com dimensões " + tensor.shapeStr()
         );
      }

      if((dimB[0] < 0 || dimB[0] >= destino.dim1()) || (dimB[1] < 0 || dimB[1] >= destino.dim2())){
         throw new IllegalArgumentException(
            "\nÍndices do tensor de destino (" + dimB[0] + ", " + dimB[1] + ") " +
            "inválidos para o tensor de destino com dimensões " + destino.shapeStr()
         );
      }

      for(int i = 0; i < destino.dim3(); i++){
         for(int j = 0; j < destino.dim4(); j++){
            destino.set(dimB[0], dimB[1], i, j, (
               tensor.get(dimA[0], dimA[1], i, j)
            ));
         }
      }
   }

   /**
    * Transpõe o conteúdo da matriz do tensor de acordo com os índices especificados.
    * @param tensor tensor desejado.
    * @param dim1 índice da primeira dimensão.
    * @param dim2 índice da segunda dimensão.
    * @return tensor transposto.
    */
   public Tensor4D matTranspor(Tensor4D tensor, int dim1, int dim2){
      Tensor4D t = new Tensor4D(1, 1, tensor.dim4(), tensor.dim3());

      for(int i = 0; i < tensor.dim3(); i++){
         for(int j = 0; j < tensor.dim4(); j++){
            t.set(0, 0, j, i, (
               tensor.get(dim1, dim2, i, j)
            ));
         }
      }

      return t;
   }

   /**
    * Realiza a adição {@code elemento a elemento} entre matrizes de tensores, como no exemplo.
    * <pre>
    *    R[i][j] = A[i][j] + B[i][j]
    * </pre>
    * @param a primeiro tensor.
    * @param b segundo tensor.
    * @param r tensor de destino do resultado.
    * @param idProfundidade índice do canal de profundidade desejado.
    */
   public void matAdd(Tensor4D a, Tensor4D b, Tensor4D r, int idProfundidade){
      int profA = a.dim2();
      int profB = b.dim2();
      int profR = r.dim2();
   
      if(
         (idProfundidade < 0 || idProfundidade >= profA) || 
         (idProfundidade < 0 || idProfundidade >= profB) || 
         (idProfundidade < 0 || idProfundidade >= profR)
         ){
         throw new IllegalArgumentException(
            "\nTodos os tensores fornecidos devem conter o índice de profundidade válido."
         );
      }

      if(compararAlturaLargura(a, b) == false){
         throw new IllegalArgumentException(
            "As dimensões de A " + a.shapeStr() + " e B " + b.shapeStr() +
            " devem ser iguais compatíveis"
         );
      }
      if(compararAlturaLargura(a, r) == false){
         throw new IllegalArgumentException(
            "As dimensões de A " + a.shapeStr() + " e R " + b.shapeStr() +
            " devem ser iguais compatíveis"
         );
      }

      r.copiar(a, 0, idProfundidade);

      int linhas = r.dim3(), colunas = r.dim4();
      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            r.add(0, idProfundidade, i, j, (
               b.get(0, idProfundidade, i, j)
            ));
         }
      }
   }

   /**
    * Realiza a adição {@code elemento a elemento} entre matrizes de tensores, como no exemplo.
    * <pre>
    *    R[i][j] = A[i][j] + B[i][j]
    * </pre>
    * @param a primeiro tensor.
    * @param b segundo tensor.
    * @return tensor com resultado.
    */
   public Tensor4D matAdd(Tensor4D a, Tensor4D b, int idProfundidade){
      int profA = a.dim2();
      int profB = b.dim2();
   
      if(
         (idProfundidade < 0 || idProfundidade >= profA) || 
         (idProfundidade < 0 || idProfundidade >= profB)
         ){
         throw new IllegalArgumentException(
            "\nTodos os tensores fornecidos devem conter o índice de profundidade válido."
         );
      }

      if(compararAlturaLargura(a, b) == false){
         throw new IllegalArgumentException(
            "As dimensões de A " + a.shapeStr() + " e B " + b.shapeStr() +
            " devem ser iguais compatíveis"
         );
      }

      int linhas = a.dim3(), colunas = b.dim4();
      Tensor4D res = new Tensor4D(1, 1, linhas, colunas);
      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            res.set(0, idProfundidade, i, j, (
               a.get(0, idProfundidade, i, j) + b.get(0, idProfundidade, i, j)
            ));
         }
      }

      return res;
   }

   /**
    * Realiza a subtração {@code elemento a elemento} entre matrizes de tensores, como no exemplo.
    * <pre>
    *    R[i][j] = A[i][j] - B[i][j]
    * </pre>
    * @param a primeiro tensor.
    * @param b segundo tensor.
    * @param r tensor de destino do resultado.
    * @param idProfundidade índice do canal de profundidade desejado.
    */
   public void matSub(Tensor4D a, Tensor4D b, Tensor4D r, int idProfundidade){
      int profA = a.dim2();
      int profB = b.dim2();
      int profR = r.dim2();
   
      if(
         (idProfundidade < 0 || idProfundidade >= profA) || 
         (idProfundidade < 0 || idProfundidade >= profB) || 
         (idProfundidade < 0 || idProfundidade >= profR)
         ){
         throw new IllegalArgumentException(
            "\nTodos os tensores fornecidos devem conter o índice de profundidade válido."
         );
      }

      if(compararAlturaLargura(a, b) == false){
         throw new IllegalArgumentException(
            "As dimensões de A " + a.shapeStr() + " e B " + b.shapeStr() +
            " devem ser iguais compatíveis"
         );
      }
      if(compararAlturaLargura(a, r) == false){
         throw new IllegalArgumentException(
            "As dimensões de A " + a.shapeStr() + " e R " + b.shapeStr() +
            " devem ser iguais compatíveis"
         );
      }

      int linhas = r.dim3(), colunas = r.dim4();
      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            r.set(0, idProfundidade, i, j, (
               a.get(0, idProfundidade, i, j) - b.get(0, idProfundidade, i, j)
            ));
         }
      }
   }

   /**
    * Realiza a subtração {@code elemento a elemento} entre matrizes de tensores, como no exemplo.
    * <pre>
    *    R[i][j] = A[i][j] - B[i][j]
    * </pre>
    * @param a primeiro tensor.
    * @param b segundo tensor.
    * @return tensor com resultado.
    */
   public Tensor4D matSub(Tensor4D a, Tensor4D b, int idProfundidade){
      int profA = a.dim2();
      int profB = b.dim2();
   
      if(
         (idProfundidade < 0 || idProfundidade >= profA) || 
         (idProfundidade < 0 || idProfundidade >= profB)
         ){
         throw new IllegalArgumentException(
            "\nTodos os tensores fornecidos devem conter o índice de profundidade válido."
         );
      }

      if(compararAlturaLargura(a, b) == false){
         throw new IllegalArgumentException(
            "As dimensões de A " + a.shapeStr() + " e B " + b.shapeStr() +
            " devem ser iguais compatíveis"
         );
      }

      int linhas = a.dim3(), colunas = b.dim4();
      Tensor4D res = new Tensor4D(1, 1, linhas, colunas);
      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            res.set(0, idProfundidade, i, j, (
               a.get(0, idProfundidade, i, j) - b.get(0, idProfundidade, i, j)
            ));
         }
      }

      return res;
   }

   /**
    * Realiza multiplicação matricial entre tensores, como no exemplo.
    * <pre>
    *    R[dim1][dim2] = A[dim1][dim2] * B[dim1][dim2]
    * </pre>
    * @param a primeiro tensor.
    * @param b segundo tensor.
    * @param r tensor de destino do resultado.
    * @param dim1 índice da primeira dimensão desejada.
    * @param dim2 índice da segunda dimensão desejada.
    */
   public void matMult(Tensor4D a, Tensor4D b, Tensor4D r, int dim1, int dim2){
      int ad1 = a.dim1(), ad2 = a.dim2();
      int bd1 = b.dim1(), bd2 = b.dim2();
      int rd1 = r.dim1(), rd2 = r.dim2();
   
      if(
         (dim1 < 0 || dim1 >= ad1) || 
         (dim1 < 0 || dim1 >= bd1) || 
         (dim1 < 0 || dim1 >= rd1)
         ){
         throw new IllegalArgumentException(
            "\nTodos os tensores fornecidos devem conter o índice de primeira dimensão válido."
         );
      }

      if(
         (dim1 < 0 || dim1 >= ad2) || 
         (dim1 < 0 || dim1 >= bd2) || 
         (dim1 < 0 || dim1 >= rd2)
         ){
         throw new IllegalArgumentException(
            "\nTodos os tensores fornecidos devem conter o índice de segunda dimensão válido."
         );
      }

      int rLin = r.dim3(), rCol = r.dim4();
      int aCol = a.dim4();
      double res = 0;

      for(int i = 0; i < rLin; i++){
         for(int j = 0; j < rCol; j++){
            res = 0;
            for(int k = 0; k < aCol; k++){
               res += a.get(dim1, dim2, i, k) * b.get(dim1, dim2, k, j);
            }
            r.set(dim1, dim2, i, j, res);
         }
      }
   }

   /**
    * Realiza multiplicação {@code elemento a elementos} entre tensores, como no exemplo.
    * <pre>
    *    R[dim1][dim2] = A[dim1][dim2] ⊙ B[dim1][dim2]
    * </pre>
    * @param a primeiro tensor.
    * @param b segundo tensor.
    * @param r tensor de destino do resultado.
    * @param dim1 índice da primeira dimensão desejada.
    * @param dim2 índice da segunda dimensão desejada.
    */
   public void matHadamard(Tensor4D a, Tensor4D b, Tensor4D r, int dim1, int dim2){
      int ad1 = a.dim1(), ad2 = a.dim2();
      int bd1 = b.dim1(), bd2 = b.dim2();
      int rd1 = r.dim1(), rd2 = r.dim2();
   
      if(
         (dim1 < 0 || dim1 >= ad1) || 
         (dim1 < 0 || dim1 >= bd1) || 
         (dim1 < 0 || dim1 >= rd1)
         ){
         throw new IllegalArgumentException(
            "\nTodos os tensores fornecidos devem conter o índice de primeira dimensão válido."
         );
      }

      if(
         (dim1 < 0 || dim1 >= ad2) || 
         (dim1 < 0 || dim1 >= bd2) || 
         (dim1 < 0 || dim1 >= rd2)
         ){
         throw new IllegalArgumentException(
            "\nTodos os tensores fornecidos devem conter o índice de segunda dimensão válido."
         );
      }
      
      if(compararAlturaLargura(a, b) == false){
         throw new IllegalArgumentException(
            "As duas últimas dimensões de A " + a.shapeStr() + " e B " + b.shapeStr() +
            " devem ser compatíveis"
         );
      }
      if(compararAlturaLargura(a, r) == false){
         throw new IllegalArgumentException(
            "As duas últimas dimensões de A " + a.shapeStr() + " e R " + b.shapeStr() +
            " devem ser compatíveis"
         );
      }

      int linhas = r.dim3(), colunas = r.dim4();
      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            r.set(dim1, dim2, i, j, (
               a.get(dim1, dim2, i, j) * b.get(dim1, dim2, i, j)
            ));
         }
      }
   }

   /**
    * Realiza multiplicação {@code elemento a elementos} entre tensores, como no exemplo.
    * <pre>
    *    R[dim1][dim2] = A[dim1][dim2] ⊙ B[dim1][dim2]
    * </pre>
    * @param a primeiro tensor.
    * @param b segundo tensor.
    * @param dim1 índice da primeira dimensão desejada.
    * @param dim2 índice da segunda dimensão desejada.
    * @return tensor de destino do resultado.
    */
   public Tensor4D matHadamard(Tensor4D a, Tensor4D b, int dim1, int dim2){
      int ad1 = a.dim1(), ad2 = a.dim2();
      int bd1 = b.dim1(), bd2 = b.dim2();
   
      if(
         (dim1 < 0 || dim1 >= ad1) || 
         (dim1 < 0 || dim1 >= bd1)
         ){
         throw new IllegalArgumentException(
            "\nTodos os tensores fornecidos devem conter o índice de primeira dimensão válido."
         );
      }

      if(
         (dim1 < 0 || dim1 >= ad2) || 
         (dim1 < 0 || dim1 >= bd2)
         ){
         throw new IllegalArgumentException(
            "\nTodos os tensores fornecidos devem conter o índice de segunda dimensão válido."
         );
      }
      
      if(compararAlturaLargura(a, b) == false){
         throw new IllegalArgumentException(
            "As duas últimas dimensões de A " + a.shapeStr() + " e B " + b.shapeStr() +
            " devem ser compatíveis"
         );
      }

      Tensor4D res = new Tensor4D(1, 1, a.dim3(), a.dim4());
      int linhas = res.dim3(), colunas = res.dim4();
      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            res.set(dim1, dim2, i, j, (
               a.get(dim1, dim2, i, j) * b.get(dim1, dim2, i, j)
            ));
         }
      }

      return res;
   }

   /**
    * Rotaciona em 180° o conteúdo da matriz contido no tensor.
    * <p>
    *    Essencialmente esse método é mais para uso de operacões convolucionais.
    * </p>
    * @param tensor tensor desejado
    * @param dim1 índice da primeira dimensão do tensor.
    * @param dim2 índice da segunda dimensão do tensor.
    * @return tensor com uma matriz rotacionada de acordo com os índices dados.
    */
   public Tensor4D rotacionarMatriz180(Tensor4D tensor, int dim1, int dim2){
      if(dim2 < 0 || dim2 > tensor.dim1()){
         throw new IllegalArgumentException(
            "Valor da dimensão 2 (" + dim2 + ") inválido."
         );
      }
      Tensor4D invertido = new Tensor4D(tensor);
      double[] arr = new double[tensor.dim3() * tensor.dim4()];
      
      int cont = 0;
      for(int i = 0; i < tensor.dim3(); i++){
         for(int j = 0; j < tensor.dim4(); j++){
            arr[cont] = tensor.get(dim1, dim2, i, j);
            cont++;
         }
      }
      oparr.inverter(arr);
      cont = 0;
      for(int i = 0; i < tensor.dim3(); i++){
         for(int j = 0; j < tensor.dim4(); j++){
            invertido.set(dim1, dim2, i, j, arr[cont]);
            cont++;
         }
      }

      return invertido;
   }

   /**
    * Realiza a operação de correlação cruzada (apenas 2D) entre dois tensores usando
    * a dimensão de profundidade desejada.
    * @param entrada tensor com a matriz de entrada.
    * @param kernel tensor com a matriz de kernel.
    * @param saida tensor de destino.
    * @param idEn índice desejado para a entrada (id[0], id[1] ...).
    * @param idK índice desejado para o kernel (id[0], id[1] ...).
    * @param idS índice desejado para a saída (id[0], id[1] ...).
    * @param add verificador para o resultado, se {@code verdadeiro} a matriz de resultados
    * não será zerada antes da operação, se {@code falso} a matriz de resultados será
    * zerada antes da operação.
    */
   public void correlacao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int[] idEn, int[] idK, int[] idS, boolean add){
      if(!entrada.validarIndice(idEn[0], 0) || !entrada.validarIndice(idEn[0], 0)){
         throw new IllegalArgumentException(
            "\nÍndices de entrada (" + idEn[0] + ", " + idEn[1] + ") " +
            "incompatíveis com o tensor de entrada (" + entrada.dim1() + ", " + entrada.dim2() + ")."
         );
      }
      if(!kernel.validarIndice(idK[0], 0) || !kernel.validarIndice(idK[0], 0)){
         throw new IllegalArgumentException(
            "\nÍndices do kernel (" + idK[0] + ", " + idK[1] + ") " +
            "incompatíveis com o tensor do kernel (" + kernel.dim1() + ", " + kernel.dim2() + ")."
         );
      }
      if(!saida.validarIndice(idS[0], 0) || !saida.validarIndice(idS[0], 0)){
         throw new IllegalArgumentException(
            "\nÍndices da saída (" + idS[0] + ", " + idS[1] + ") " +
            "incompatíveis com o tensor de saída (" + saida.dim1() + ", " + saida.dim2() + ")."
         );
      }

      if(!add){
         saida.preencher2D(idS[0], idS[1] ,0.0);
      }

      int alturaEsperada  = entrada.dim3() - kernel.dim3() + 1;
      int larguraEsperada = entrada.dim4() - kernel.dim4() + 1;
      if(saida.dim3() != alturaEsperada){
         throw new IllegalArgumentException(
            "\nAltura da saída (" + saida.dim3() + 
            ") íncompatível com o valor esperado (" + alturaEsperada + ")."
         );
      }
      if(saida.dim4() != larguraEsperada){
         throw new IllegalArgumentException(
            "\nAltura da saída (" + saida.dim4() + 
            ") íncompatível com o valor esperado (" + larguraEsperada + ")."
         );
      }

      int alturaKernel = kernel.dim3();
      int larguraKernel = kernel.dim4();
      double soma = 0d;
      int i, j, m, n;
      int posX;
      for(i = 0; i < alturaEsperada; i++){
         for(j = 0; j < larguraEsperada; j++){
            soma = 0.d;
            for(m = 0; m < alturaKernel; m++){
               posX = i + m;
               for(n = 0; n < larguraKernel; n++){
                  soma += entrada.get(idEn[0], idEn[1], posX, j+n) * 
                           kernel.get(idK[0], idK[1], m, n);
               }
            }
            saida.add(idS[0], idS[1], i, j, soma);
         }
      }
   }

   /**
    * Realiza a operação de correlação cruzada (apenas 2D) entre dois tensores usando
    * a dimensão primeira dimensão dos tensores e a segunda dimensão (profundidade) desejada.
    * @param entrada tensor com a matriz de entrada.
    * @param kernel tensor com a matriz de kernel.
    * @param saida tensor de destino.
    * @param idProfundidade índice do canal de profundidade desejado.
    */
   public void correlacao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int idProfundidade){
      correlacao2D(
         entrada, 
         kernel, 
         saida, 
         new int[]{0, idProfundidade}, 
         new int[]{0, idProfundidade}, 
         new int[]{0, idProfundidade}, 
         false
      );
   }

   /**
    * Realiza a operação de correlação cruzada (apenas 2D) entre dois tensores usando
    * as duas primeiras dimensões de profundidade.
    * @param entrada tensor com a matriz de entrada.
    * @param kernel tensor com a matriz de kernel.
    * @param saida tensor de destino.
    */
   public void correlacao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida){
      correlacao2D(
         entrada, 
         kernel, 
         saida, 
         new int[]{0, 0}, 
         new int[]{0, 0}, 
         new int[]{0, 0}, 
         false
      );
   }

   /**
    * Realiza a operação de convolução (apenas 2D) entre dois tensores usando
    * a dimensão de profundidade desejada.
    * @param entrada tensor com a matriz de entrada.
    * @param kernel tensor com a matriz de kernel.
    * @param saida tensor de destino.
    * @param idEn índice desejado para a entrada (id[0], id[1] ...).
    * @param idK índice desejado para o kernel (id[0], id[1] ...).
    * @param idS índice desejado para a saída (id[0], id[1] ...).
    * @param add verificador para o resultado, se {@code verdadeiro} a matriz de resultados
    * não será zerada antes da operação, se {@code falso} a matriz de resultados será
    * zerada antes da operação.
    */
   public void convolucao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int[] idEn, int[] idK, int[] idS, boolean add){
      if(!entrada.validarIndice(idEn[0], 0) || !entrada.validarIndice(idEn[0], 0)){
         throw new IllegalArgumentException(
            "\nÍndices de entrada (" + idEn[0] + ", " + idEn[1] + ") " +
            "incompatíveis com o tensor de entrada (" + entrada.dim1() + ", " + entrada.dim2() + ")."
         );
      }
      if(!kernel.validarIndice(idK[0], 0) || !kernel.validarIndice(idK[0], 0)){
         throw new IllegalArgumentException(
            "\nÍndices do kernel (" + idEn[0] + ", " + idEn[1] + ") " +
            "incompatíveis com o tensor do kernel (" + entrada.dim1() + ", " + entrada.dim2() + ")."
         );
      }
      if(!saida.validarIndice(idS[0], 0) || !saida.validarIndice(idS[0], 0)){
         throw new IllegalArgumentException(
            "\nÍndices da saída (" + idEn[0] + ", " + idEn[1] + ") " +
            "incompatíveis com o tensor de saída (" + entrada.dim1() + ", " + entrada.dim2() + ")."
         );
      }

      if(!add){
         saida.preencher2D(idS[0], idS[1] ,0);
      }

      int alturaEsperada  = entrada.dim3() - kernel.dim3() + 1;
      int larguraEsperada = entrada.dim4() - kernel.dim4() + 1;
      if(saida.dim3() != alturaEsperada){
         throw new IllegalArgumentException(
            "\nAltura da saída (" + saida.dim3() + 
            ") íncompatível com o valor esperado (" + alturaEsperada + ")."
         );
      }
      if(saida.dim4() != larguraEsperada){
         throw new IllegalArgumentException(
            "\nAltura da saída (" + saida.dim4() + 
            ") íncompatível com o valor esperado (" + larguraEsperada + ")."
         );
      }

      Tensor4D rotacionado = rotacionarMatriz180(kernel, idK[0], idK[1]);

      int alturaKernel = rotacionado.dim3();
      int larguraKernel = rotacionado.dim4();
      int i, j, m, n;
      double soma;
      int posX, posY;
      for(i = 0; i < alturaEsperada; i++){
         for(j = 0; j < larguraEsperada; j++){
            soma = 0.0;
            for(m = 0; m < alturaKernel; m++){
               posX = i + m;
               for(n = 0; n < larguraKernel; n++){
                  posY = j + n;
                  soma += entrada.get(idEn[0], idEn[1], posX, posY) * 
                        rotacionado.get(idK[0], idK[1], m, n);
               }
            }
            saida.add(idS[0], idS[1], i, j, soma);
         }
      }
   }

   /**
    * Realiza a operação de convolução (apenas 2D) entre dois tensores usando
    * a dimensão primeira dimensão dos tensores e a segunda dimensão (profundidade) desejada.
    * @param entrada tensor com a matriz de entrada.
    * @param kernel tensor com a matriz de kernel.
    * @param saida tensor de destino.
    * @param idProfundidade índice do canal de profundidade desejado.
    */
   public void convolucao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int idProfundidade){
      convolucao2D(
         entrada, 
         kernel, 
         saida, 
         new int[]{0, idProfundidade}, 
         new int[]{0, idProfundidade}, 
         new int[]{0, idProfundidade}, 
         false
      );
   }

   /**
    * Realiza a operação de correlação convolução (apenas 2D) entre dois tensores usando
    * a primeira dimensão de profundidade.
    * @param entrada tensor com a matriz de entrada.
    * @param kernel tensor com a matriz de kernel.
    * @param saida tensor de destino.
    */
   public void convolucao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida){
      convolucao2D(
         entrada, 
         kernel, 
         saida, 
         new int[]{0, 0}, 
         new int[]{0, 0}, 
         new int[]{0, 0}, 
         false
      );
   }

   /**
    * Realiza a operação de convolução (apenas 2D) entre dois tensores usando
    * a dimensão de profundidade desejada.
    * @param entrada tensor com a matriz de entrada.
    * @param kernel tensor com a matriz de kernel.
    * @param saida tensor de destino.
    * @param idEn índice desejado para a entrada (id[0], id[1] ...).
    * @param idK índice desejado para o kernel (id[0], id[1] ...).
    * @param idS índice desejado para a saída (id[0], id[1] ...).
    * @param add verificador para o resultado, se {@code verdadeiro} a matriz de resultados
    * não será zerada antes da operação, se {@code falso} a matriz de resultados será
    * zerada antes da operação.
    */
   public void convolucao2DFull(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int[] idEn, int[] idK, int[] idS, boolean add){
      if(!entrada.validarIndice(idEn[0], 0) || !entrada.validarIndice(idEn[0], 0)){
         throw new IllegalArgumentException(
            "\nÍndices de entrada (" + idEn[0] + ", " + idEn[1] + ") " +
            "incompatíveis com o tensor de entrada (" + entrada.dim1() + ", " + entrada.dim2() + ")."
         );
      }
      if(!kernel.validarIndice(idK[0], 0) || !kernel.validarIndice(idK[0], 0)){
         throw new IllegalArgumentException(
            "\nÍndices do kernel (" + idEn[0] + ", " + idEn[1] + ") " +
            "incompatíveis com o tensor do kernel (" + entrada.dim1() + ", " + entrada.dim2() + ")."
         );
      }
      if(!saida.validarIndice(idS[0], 0) || !saida.validarIndice(idS[0], 0)){
         throw new IllegalArgumentException(
            "\nÍndices da saída (" + idEn[0] + ", " + idEn[1] + ") " +
            "incompatíveis com o tensor de saída (" + entrada.dim1() + ", " + entrada.dim2() + ")."
         );
      }

      if(!add){
         saida.preencher2D(idS[0], idS[1], 0);
      }

      int alturaEsperada  = entrada.dim3() + kernel.dim3() - 1;
      int larguraEsperada = entrada.dim4() + kernel.dim4() - 1;
      if(saida.dim3() != alturaEsperada){
         throw new IllegalArgumentException(
            "\nAltura da saída (" + saida.dim3() + 
            ") íncompatível com o valor esperado (" + alturaEsperada + ")."
         );
      }
      if(saida.dim4() != larguraEsperada){
         throw new IllegalArgumentException(
            "\nAltura da saída (" + saida.dim4() + 
            ") íncompatível com o valor esperado (" + larguraEsperada + ")."
         );
      }

      int i, j, k, l, posX, posY;
      double res;
      int linEntrada = entrada.dim3(), colEntrada = entrada.dim4();
      int linKernel = kernel.dim3(), colKernel = kernel.dim4();
      int linSaida = saida.dim3(), colSaida = saida.dim4();
      for(i = 0; i < linSaida; i++){
         for(j = 0; j < colSaida; j++){
            res = 0;
            for(k = 0; k < linKernel; k++){
               posX = i - k;
               for(l = 0; l < colKernel; l++){
                  posY = j - l;
  
                  if(posX >= 0 && posX < linEntrada && posY >= 0 && posY < colEntrada){
                     res += 
                        entrada.get(idEn[0], idEn[1], posX, posY) * 
                        kernel.get(idK[0], idK[1], k, l);
                  }
               }
            }
            saida.add(idS[0], idS[1], i, j, res);
         }
      }
   }

   /**
    * Método exluviso para a propagação direta de camadas convolucionais
    * @param entrada tensor de entrada.
    * @param kernel tensor dos kernels.
    * @param saida tensor de destino.
    */
   public void convForward(Tensor4D entrada, Tensor4D kernel, Tensor4D saida){
      int filtros = kernel.dim1();
      int entradas = kernel.dim2();

      int[] idSaida = {0, 0};
      int[] idEntrada = {0, 0};
      int[] idKernel = {0, 0};
      for(int i = 0; i < filtros; i++){
         idSaida[1] = i;
         idKernel[0] = i;
         for(int j = 0; j < entradas; j++){
            idEntrada[1] = j;
            idKernel[1] = j;
            correlacao2D(entrada, kernel, saida, idEntrada, idKernel, idSaida, true);
         }
      }
   }
   
   /**
    * Método exluviso para a propagação reversa de camadas convolucionais.
    * @param entrada tensor de entrada da camada.
    * @param kernel tensor dos kernels.
    * @param gradSaida tensor com os valores dos gradientes da camada em relação a sua saída.
    * @param gradK tensor dos gradientes em relação aos filtros.
    * @param gradE tensor com o gradiente de entrada.
    */
   public void convBackward(Tensor4D entrada, Tensor4D kernel, Tensor4D gradSaida, Tensor4D gradK, Tensor4D gradE){
      int filtros = kernel.dim1();
      int entradas = kernel.dim2();
  
      int[] idEn = {0, 0};
      int[] idGradSaida = {0, 0};
      int[] idK = {0, 0};
      int[] idGradEn = {0, 0};
      for(int i = 0; i < filtros; i++){
         idGradSaida[1] = i;
         idK[0] = i;
         for(int j = 0; j < entradas; j++){
            idEn[1] = j;
            idK[1] = j;
            idGradEn[1] = j;
            correlacao2D(entrada, gradSaida, gradK, idEn, idGradSaida, idK, false);
            convolucao2DFull(gradSaida, kernel, gradE, idGradSaida, idK, idGradEn, true);
         }
      }

   }
}
