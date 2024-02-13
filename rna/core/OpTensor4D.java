package rna.core;

public class OpTensor4D{

   OpArray oparr = new OpArray();
   
   public OpTensor4D(){}

   public Tensor4D inverterMatriz(Tensor4D tensor, int dim1, int dim2){
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

      Tensor4D rotacionado = inverterMatriz(kernel, 0, idProfundidade);
  
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
}
