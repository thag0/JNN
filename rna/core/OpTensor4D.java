package rna.core;

public class OpTensor4D{

   OpArray oparr = new OpArray();
   
   public OpTensor4D(){}

   /**
    * Verifica se os tensores possuem a mesma altura (dim3) e largura (dim4).
    * @param a tensor A.
    * @param b tensor B.
    * @return resultado da verificação.
    */
   public boolean compararAlturaLargura(Tensor4D a, Tensor4D b){
      if((a.dim3() != b.dim3()) || (a.dim4() != b.dim4())){
         return false;
      }

      return true;
   }


   /**
    * 
    * @param tensor
    * @param idProfundidade
    */
   public void copiarMatriz(Tensor4D a, Tensor4D r, int idProfundidade){
      for(int i = 0; i < r.dim3(); i++){
         for(int j = 0; j < r.dim4(); j++){
            r.editar(0, idProfundidade, i, j, (
               a.elemento(0, idProfundidade, i, j)
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
      Tensor4D t = new Tensor4D(tensor.dim1(), tensor.dim2(), tensor.dim4(), tensor.dim3());

      for(int i = 0; i < tensor.dim3(); i++){
         for(int j = 0; j < tensor.dim4(); j++){
            t.editar(dim1, dim2, j, i, (
               tensor.elemento(dim1, dim2, i, j)
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
            "As dimensões de A " + a.dimensoesStr() + " e B " + b.dimensoesStr() +
            " devem ser iguais compatíveis"
         );
      }
      if(compararAlturaLargura(a, r) == false){
         throw new IllegalArgumentException(
            "As dimensões de A " + a.dimensoesStr() + " e R " + b.dimensoesStr() +
            " devem ser iguais compatíveis"
         );
      }

      int linhas = r.dim3(), colunas = r.dim4();
      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            r.editar(0, idProfundidade, i, j, (
               a.elemento(0, idProfundidade, i, j) + b.elemento(0, idProfundidade, i, j)
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
            "As dimensões de A " + a.dimensoesStr() + " e B " + b.dimensoesStr() +
            " devem ser iguais compatíveis"
         );
      }

      int linhas = a.dim3(), colunas = b.dim4();
      Tensor4D res = new Tensor4D(1, 1, linhas, colunas);
      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            res.editar(0, idProfundidade, i, j, (
               a.elemento(0, idProfundidade, i, j) + b.elemento(0, idProfundidade, i, j)
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
            "As dimensões de A " + a.dimensoesStr() + " e B " + b.dimensoesStr() +
            " devem ser iguais compatíveis"
         );
      }
      if(compararAlturaLargura(a, r) == false){
         throw new IllegalArgumentException(
            "As dimensões de A " + a.dimensoesStr() + " e R " + b.dimensoesStr() +
            " devem ser iguais compatíveis"
         );
      }

      int linhas = r.dim3(), colunas = r.dim4();
      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            r.editar(0, idProfundidade, i, j, (
               a.elemento(0, idProfundidade, i, j) - b.elemento(0, idProfundidade, i, j)
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
            "As dimensões de A " + a.dimensoesStr() + " e B " + b.dimensoesStr() +
            " devem ser iguais compatíveis"
         );
      }

      int linhas = a.dim3(), colunas = b.dim4();
      Tensor4D res = new Tensor4D(1, 1, linhas, colunas);
      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            res.editar(0, idProfundidade, i, j, (
               a.elemento(0, idProfundidade, i, j) - b.elemento(0, idProfundidade, i, j)
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
               res += a.elemento(dim1, dim2, i, k) * b.elemento(dim1, dim2, k, j);
            }
            r.editar(dim1, dim2, i, j, res);
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
            "As duas últimas dimensões de A " + a.dimensoesStr() + " e B " + b.dimensoesStr() +
            " devem ser compatíveis"
         );
      }
      if(compararAlturaLargura(a, r) == false){
         throw new IllegalArgumentException(
            "As duas últimas dimensões de A " + a.dimensoesStr() + " e R " + b.dimensoesStr() +
            " devem ser compatíveis"
         );
      }

      System.out.println(a.nome() + " " + a.dimensoesStr());
      System.out.println(b.nome() + " " + b.dimensoesStr());
      System.out.println(r.nome() + " " + r.dimensoesStr());

      int linhas = r.dim3(), colunas = r.dim4();
      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            r.editar(dim1, dim2, i, j, (
               a.elemento(dim1, dim2, i, j) * b.elemento(dim1, dim2, i, j)
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
            "As duas últimas dimensões de A " + a.dimensoesStr() + " e B " + b.dimensoesStr() +
            " devem ser compatíveis"
         );
      }

      Tensor4D res = new Tensor4D(1, 1, a.dim3(), a.dim4());
      int linhas = res.dim3(), colunas = res.dim4();
      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            res.editar(dim1, dim2, i, j, (
               a.elemento(dim1, dim2, i, j) * b.elemento(dim1, dim2, i, j)
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
            arr[cont] = tensor.elemento(dim1, dim2, i, j);
            cont++;
         }
      }
      oparr.inverter(arr);
      cont = 0;
      for(int i = 0; i < tensor.dim3(); i++){
         for(int j = 0; j < tensor.dim4(); j++){
            invertido.editar(dim1, dim2, i, j, arr[cont]);
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
    * @param idEntrada índice desejado para a entrada (id[0], id[1] ...).
    * @param idKernel índice desejado para o kernel (id[0], id[1] ...).
    * @param idSaida índice desejado para a saída (id[0], id[1] ...).
    * @param add verificador para o resultado, se {@code verdadeiro} a matriz de resultados
    * não será zerada antes da operação, se {@code falso} a matriz de resultados será
    * zerada antes da operação.
    */
   public void correlacao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int[] idEntrada, int[] idKernel, int[] idSaida, boolean add){
      if((idEntrada[0] < 0 || idEntrada[0] >= entrada.dim1()) || (idEntrada[1] < 0 || idEntrada[1] >= entrada.dim2())){
         throw new IllegalArgumentException(
            "\nÍndices de entrada (" + idEntrada[0] + ", " + idEntrada[1] + ") " +
            "incompatíveis com o tensor de entrada (" + entrada.dim1() + ", " + entrada.dim2() + ")."
         );
      }
      if((idKernel[0] < 0 || idKernel[0] >= kernel.dim1()) || (idKernel[1] < 0 || idKernel[1] >= kernel.dim2())){
         throw new IllegalArgumentException(
            "\nÍndices do kernel (" + idKernel[0] + ", " + idKernel[1] + ") " +
            "incompatíveis com o tensor do kernel (" + kernel.dim1() + ", " + kernel.dim2() + ")."
         );
      }
      if((idSaida[0] < 0 || idSaida[0] >= saida.dim1()) || (idSaida[1] < 0 || idSaida[1] >= saida.dim2())){
         throw new IllegalArgumentException(
            "\nÍndices da saída (" + idSaida[0] + ", " + idSaida[1] + ") " +
            "incompatíveis com o tensor de saída (" + saida.dim1() + ", " + saida.dim2() + ")."
         );
      }

      if(add == false){
         saida.preencher2D(idSaida[0], idSaida[1] ,0);
      }

      int alturaEsperada = entrada.dim3()-kernel.dim3()+1;
      int larguraEsperada = entrada.dim4()-kernel.dim4()+1;
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
      for(int i = 0; i < alturaEsperada; i++){
         for(int j = 0; j < larguraEsperada; j++){
            double soma = 0.0;
            for(int m = 0; m < alturaKernel; m++){
                 for(int n = 0; n < larguraKernel; n++){
                  int posX = i + m;
                  int posY = j + n;
                  soma += entrada.elemento(idEntrada[0], idEntrada[1], posX, posY) * 
                           kernel.elemento(idKernel[0], idKernel[1], m, n);
               }
            }
            saida.add(idSaida[0], idSaida[1], i, j, soma);
         }
      }
   }

   /**
    * Realiza a operação de correlação cruzada (apenas 2D) entre dois tensores usando
    * a dimensão de profundidade desejada.
    * @param entrada tensor com a matriz de entrada.
    * @param kernel tensor com a matriz de kernel.
    * @param saida tensor de destino.
    * @param idProfundidade índice do canal de profundidade desejado.
    */
   public void correlacao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int idProfundidade){
      if(idProfundidade < 0 || idProfundidade >= entrada.dim2()){
         throw new IllegalArgumentException(
            "\nO tensor de entrada " + entrada.dimensoesStr() + 
            " não possui a profundidade desejada ("+ idProfundidade + ")."
         );
      }
      if(idProfundidade < 0 || idProfundidade >= kernel.dim2()){
         throw new IllegalArgumentException(
            "\nO tensor de kernel " + entrada.dimensoesStr() + 
            " não possui a profundidade desejada ("+ idProfundidade + ")."
         );
      }

      int alturaEsperada = entrada.dim3()-kernel.dim3()+1;
      int larguraEsperada = entrada.dim4()-kernel.dim4()+1;
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
      for(int i = 0; i < alturaEsperada; i++){
         for(int j = 0; j < larguraEsperada; j++){
            double soma = 0.0;
            for(int m = 0; m < alturaKernel; m++){
                 for(int n = 0; n < larguraKernel; n++){
                  int posX = i + m;
                  int posY = j + n;
                  soma += entrada.elemento(0, idProfundidade, posX, posY) * 
                           kernel.elemento(0, idProfundidade, m, n);
               }
             }
            saida.editar(0, idProfundidade, i, j, soma);
         }
      }
   }

   /**
    * Realiza a operação de correlação cruzada (apenas 2D) entre dois tensores usando
    * a primeira dimensão de profundidade.
    * @param entrada tensor com a matriz de entrada.
    * @param kernel tensor com a matriz de kernel.
    * @param saida tensor de destino.
    */
   public void correlacao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida){
      correlacao2D(entrada, kernel, saida, 0);
   }

   /**
    * Realiza a operação de convolução (apenas 2D) entre dois tensores usando
    * a dimensão de profundidade desejada.
    * @param entrada tensor com a matriz de entrada.
    * @param kernel tensor com a matriz de kernel.
    * @param saida tensor de destino.
    * @param idEntrada índice desejado para a entrada (id[0], id[1] ...).
    * @param idKernel índice desejado para o kernel (id[0], id[1] ...).
    * @param idSaida índice desejado para a saída (id[0], id[1] ...).
    * @param add verificador para o resultado, se {@code verdadeiro} a matriz de resultados
    * não será zerada antes da operação, se {@code falso} a matriz de resultados será
    * zerada antes da operação.
    */
   public void convolucao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int[] idEntrada, int[] idKernel, int[] idSaida, boolean add){
      if((idEntrada[0] < 0 || idEntrada[0] >= entrada.dim1()) || (idEntrada[1] < 0 || idEntrada[1] >= entrada.dim2())){
         throw new IllegalArgumentException(
            "\nÍndices de entrada (" + idEntrada[0] + ", " + idEntrada[1] + ") " +
            "incompatíveis com o tensor de entrada (" + entrada.dim1() + ", " + entrada.dim2() + ")."
         );
      }
      if((idKernel[0] < 0 || idKernel[0] >= kernel.dim1()) || (idKernel[1] < 0 || idKernel[1] >= kernel.dim2())){
         throw new IllegalArgumentException(
            "\nÍndices do kernel (" + idEntrada[0] + ", " + idEntrada[1] + ") " +
            "incompatíveis com o tensor do kernel (" + entrada.dim1() + ", " + entrada.dim2() + ")."
         );
      }
      if((idSaida[0] < 0 || idSaida[0] >= saida.dim1()) || (idSaida[1] < 0 || idSaida[1] >= saida.dim2())){
         throw new IllegalArgumentException(
            "\nÍndices da saída (" + idEntrada[0] + ", " + idEntrada[1] + ") " +
            "incompatíveis com o tensor de saída (" + entrada.dim1() + ", " + entrada.dim2() + ")."
         );
      }

      if(add == false){
         saida.preencher2D(idSaida[0], idSaida[1] ,0);
      }

      int alturaEsperada = entrada.dim3()-kernel.dim3()+1;
      int larguraEsperada = entrada.dim4()-kernel.dim4()+1;
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

      Tensor4D rotacionado = rotacionarMatriz180(kernel, idKernel[0], idKernel[1]);

      int alturaKernel = rotacionado.dim3();
      int larguraKernel = rotacionado.dim4();
      for(int i = 0; i < alturaEsperada; i++){
         for(int j = 0; j < larguraEsperada; j++){
            double soma = 0.0;
            for(int m = 0; m < alturaKernel; m++){
                 for(int n = 0; n < larguraKernel; n++){
                  int posX = i + m;
                  int posY = j + n;
                  soma += entrada.elemento(idEntrada[0], idEntrada[1], posX, posY) * 
                        rotacionado.elemento(idKernel[0], idKernel[1], m, n);
               }
            }
            saida.add(idSaida[0], idSaida[1], i, j, soma);
         }
      }
   }

   /**
    * Realiza a operação de convolução (apenas 2D) entre dois tensores usando
    * a dimensão de profundidade desejada.
    * @param entrada tensor com a matriz de entrada.
    * @param kernel tensor com a matriz de kernel.
    * @param saida tensor de destino.
    * @param idEntrada índice desejado para a entrada (id[0], id[1] ...).
    * @param idKernel índice desejado para o kernel (id[0], id[1] ...).
    * @param idSaida índice desejado para a saída (id[0], id[1] ...).
    * @param add verificador para o resultado, se {@code verdadeiro} a matriz de resultados
    * não será zerada antes da operação, se {@code falso} a matriz de resultados será
    * zerada antes da operação.
    */
   public void convolucao2DFull(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int[] idEntrada, int[] idKernel, int[] idSaida, boolean add){
      if((idEntrada[0] < 0 || idEntrada[0] >= entrada.dim1()) || (idEntrada[1] < 0 || idEntrada[1] >= entrada.dim2())){
         throw new IllegalArgumentException(
            "\nÍndices de entrada (" + idEntrada[0] + ", " + idEntrada[1] + ") " +
            "incompatíveis com o tensor de entrada (" + entrada.dim1() + ", " + entrada.dim2() + ")."
         );
      }
      if((idKernel[0] < 0 || idKernel[0] >= kernel.dim1()) || (idKernel[1] < 0 || idKernel[1] >= kernel.dim2())){
         throw new IllegalArgumentException(
            "\nÍndices do kernel (" + idKernel[0] + ", " + idKernel[1] + ") " +
            "incompatíveis com o tensor do kernel (" + kernel.dim1() + ", " + kernel.dim2() + ")."
         );
      }
      if((idSaida[0] < 0 || idSaida[0] >= saida.dim1()) || (idSaida[1] < 0 || idSaida[1] >= saida.dim2())){
         throw new IllegalArgumentException(
            "\nÍndices da saída (" + idSaida[0] + ", " + idSaida[1] + ") " +
            "incompatíveis com o tensor de saída (" + saida.dim1() + ", " + saida.dim2() + ")."
         );
      }

      if(add == false){
         saida.preencher2D(idSaida[0], idSaida[1] ,0);
      }

      int alturaEsperada = entrada.dim3()+kernel.dim3()-1;
      int larguraEsperada = entrada.dim4()+kernel.dim4()-1;
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
               for(l = 0; l < colKernel; l++){
                  posX = i - k;
                  posY = j - l;
  
                  if(posX >= 0 && posX < linEntrada && posY >= 0 && posY < colEntrada){
                     res += 
                        entrada.elemento(idEntrada[0], idEntrada[1], posX, posY) * kernel.elemento(idKernel[0], idKernel[1], k, l);
                  }
               }
            }
            saida.add(idSaida[0], idSaida[1], i, j, res);
         }
      }
   }

   /**
    * Realiza a operação de correlação convolução (apenas 2D) entre dois tensores usando
    * a dimensão de profundidade desejada.
    * @param entrada tensor com a matriz de entrada.
    * @param kernel tensor com a matriz de kernel.
    * @param saida tensor de destino.
    * @param idProfundidade índice do canal de profundidade desejado.
    */
   public void convolucao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida, int idProfundidade){
      int alturaEsperada = entrada.dim3() - kernel.dim3() + 1;
      int larguraEsperada = entrada.dim4() - kernel.dim4() + 1;
      
      if(saida.dim3() != alturaEsperada){
         throw new IllegalArgumentException(
            "\nAltura da saída (" + saida.dim3() + ") incompatível com o valor esperado (" + alturaEsperada + ")."
         );
      }
      
      if(saida.dim4() != larguraEsperada){
         throw new IllegalArgumentException(
            "\nLargura da saída (" + saida.dim4() + ") incompatível com o valor esperado (" + larguraEsperada + ")."
         );
      }
  
      int alturaKernel = kernel.dim3();
      int larguraKernel = kernel.dim4();

      Tensor4D rotacionado = rotacionarMatriz180(kernel, 0, idProfundidade);
  
      for(int i = 0; i < alturaEsperada; i++){
         for(int j = 0; j < larguraEsperada; j++){
            double soma = 0.0;
            for(int m = 0; m < alturaKernel; m++){
                 for(int n = 0; n < larguraKernel; n++){
                  int posX = i + m;
                  int posY = j + n;
                  soma += entrada.elemento(0, idProfundidade, posX, posY) * 
                           rotacionado.elemento(0, idProfundidade, m, n);
               }
             }
            saida.editar(0, idProfundidade, i, j, soma);
         }
      }
   }

   /**
    * Realiza a operação de correlação convolução (apenas 2D) entre dois tensores usando
    * a primeira dimensão de profundidade.
    * @param entrada tensor com a matriz de entrada.
    * @param kernel tensor com a matriz de kernel.
    * @param saida tensor de destino.
    */
   public void convolucao2D(Tensor4D entrada, Tensor4D kernel, Tensor4D saida){
      convolucao2D(entrada, kernel, saida, 0);
   }

   /**
    * Método exluviso para a propagação direta de camadas convolucionais
    * @param entrada tensor de entrada.
    * @param kernel tensor dos kernels.
    * @param saida tensor de destino.
    */
   public void convForward(Tensor4D entrada, Tensor4D kernel, Tensor4D saida){
      int numFiltros = kernel.dim1();
      int profEntrada = kernel.dim2();

      for(int i = 0; i < numFiltros; i++){
         int[] idSaida = {0, i};
         for(int j = 0; j < profEntrada; j++){
            int[] idEntrada = {0, j};
            int[] idKernel = {i, j};
            correlacao2D(entrada, kernel, saida, idEntrada, idKernel, idSaida, true);
         }
      }
   }

   /**
    * Método exluviso para a propagação reversa de camadas convolucionais
    * @param entrada tensor de entrada.
    * @param kernel tensor dos kernels.
    * @param saida tensor de destino.
    */
   public void convBackward(Tensor4D entrada, Tensor4D kernel, Tensor4D derivada, Tensor4D gradKernel, Tensor4D gradEntrada){
      int numFiltros = kernel.dim1();
      int profEntrada = kernel.dim2();

      for(int i = 0; i < numFiltros; i++){
         for(int j = 0; j < profEntrada; j++){
            int[] idEntrada = {0, j};
            int[] idDerivada = {0, i};
            int[] idGradKernel = {i, j};
            correlacao2D(entrada, derivada, gradKernel, idEntrada, idDerivada, idGradKernel, false);
         }
      }

      for(int i = 0; i < numFiltros; i++){
         for(int j = 0; j < profEntrada; j++){
            int[] idDerivada = {0, i};
            int[] idKernel = {i, j};
            int[] idGradEntrada = {0, j};
            convolucao2DFull(derivada, kernel, gradEntrada, idDerivada, idKernel, idGradEntrada, true);
         }
      }

   }
}
