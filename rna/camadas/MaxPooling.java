package rna.camadas;

import rna.core.Mat;
import rna.core.Utils;
import rna.inicializadores.Inicializador;

/**
 * <h2>
 *    Camada de agrupamento máximo
 * </h2>
 * <p>
 *    A camada de agrupamento máximo é um componente utilizado para reduzir a 
 *    dimensionalidade espacial dos dados, preservando as características mais 
 *    importantes para a saída.
 * </p>
 * <p>
 *    Durante a operação de agrupamento máximo, a entrada é dividida em regiões 
 *    menores usando uma márcara e o valor máximo de cada região é salvo. 
 *    Essencialmente, a camada realiza a operação de subamostragem, mantendo apenas 
 *    as informações mais relevantes.
 * </p>
 * Exemplo simples de operação Max Pooling para uma região 2x2:
 * <pre>
 *entrada = [
 *    1, 2
 *    3, 4
 *]
 * 
 *saida = 4
 * </pre>
 * <p>
 *    A camada de max pooling não possui parâmetros treináveis nem função de ativação.
 * </p>
 */
public class MaxPooling extends Camada{

   /**
    * Utilitario.
    */
   Utils utils = new Utils();

   /**
    * Dimensões dos dados de entrada (altura, largura, profundidade)
    */
   private int[] formEntrada;

   /**
    * Dimensões dos dados de saída (altura, largura, profundidade)
    */
   private int[] formSaida;

   /**
    * Array de matrizes contendo os dados de entrada.
    */
   private Mat[] entrada;

   /**
    * Array de matrizes contendo os dados de saída.
    */
   private Mat[] saida;

   /**
    * Array de matrizes contendo os gradientes que serão
    * retropropagados para as camadas anteriores.
    */
   private Mat[] gradEntrada;

   /**
    * Formato do filtro de pooling (altura, largura).
    */
   private int[] formFiltro;

   /**
    * Valores de stride (altura, largura).
    */
   private int[] stride;

   /**
    * Instancia uma nova camada de max pooling, definindo o formato do
    * filtro que será aplicado em cada entrada da camada.
    * <p>
    *    O formato do filtro deve conter as dimensões da entrada da
    *    camada (altura, largura).
    * </p>
    * <p>
    *    Por padrão, os valores de strides serão os mesmo usados para
    *    as dimensões do filtro, exemplo:
    * </p>
    * <pre>
    *filtro = (2, 2)
    *stride = (2, 2) // valor padrão
    * </pre>
    * @param formFiltro formato do filtro de max pooling.
    * @throws IllegalArgumentException se o formato do filtro não atender as
    * requisições.
    */
   public MaxPooling(int[] formFiltro){
      if(formFiltro == null){
         throw new IllegalArgumentException(
            "O formato do filtro não pode ser nulo."
         );
      }
      if(formFiltro.length != 2){
         throw new IllegalArgumentException(
            "O formato do filtro deve conter dois elementos (altura, largura)."
         );
      }
      if(utils.apenasMaiorZero(formFiltro) == false){
         throw new IllegalArgumentException(
            "Os valores de dimensões do filtro devem ser maiores que zero."
         );
      }

      this.formFiltro = formFiltro;
      this.stride = new int[]{
         formFiltro[0],
         formFiltro[1]
      };
   }

   /**
    * Instancia uma nova camada de max pooling, definindo o formato do filtro 
    * e os strides (passos) que serão aplicados em cada entrada da camada.
    * <p>
    *    O formato do filtro e dos strides devem conter as dimensões da entrada 
    *    da camada (altura, largura).
    * </p>
    * @param formFiltro formato do filtro de max pooling.
    * @param stride strides que serão aplicados ao filtro.
    * @throws IllegalArgumentException se o formato do filtro não atender as
    * requisições.
    * @throws IllegalArgumentException se os strides não atenderem as requisições.
    */
   public MaxPooling(int[] formFiltro, int[] stride){
      if(formFiltro == null){
         throw new IllegalArgumentException(
            "O formato do filtro não pode ser nulo."
         );
      }
      if(formFiltro.length != 2){
         throw new IllegalArgumentException(
            "O formato do filtro deve conter três elementos (altura, largura)."
         );
      }
      if(utils.apenasMaiorZero(formFiltro) == false){
         throw new IllegalArgumentException(
            "Os valores de dimensões do filtro devem ser maiores que zero."
         );
      }

      if(stride == null){
         throw new IllegalArgumentException(
            "O formato do filtro não pode ser nulo."
         );
      }
      if(stride.length != 2){
         throw new IllegalArgumentException(
            "O formato para os strides deve conter dois elementos (altura, largura)."
         );
      }
      if(utils.apenasMaiorZero(stride) == false){
         throw new IllegalArgumentException(
            "Os valores para os strides devem ser maiores que zero."
         );
      }

      this.formFiltro = formFiltro;
      this.stride = stride;
   }

   /**
    * Instancia uma nova camada de max pooling, definindo o formato do filtro, 
    * formato de entrada e os strides (passos) que serão aplicados em cada entrada 
    * da camada.
    * <p>
    *    O formato do filtro e dos strides devem conter as dimensões da entrada 
    *    da camada (altura, largura).
    * </p>
    * A camada será automaticamente construída usando o formato de entrada especificado.
    * @param formEntrada formato de entrada para a camada.
    * @param formFiltro formato do filtro de max pooling.
    * @param stride strides que serão aplicados ao filtro.
    * @throws IllegalArgumentException se o formato do filtro não atender as
    * requisições.
    * @throws IllegalArgumentException se os strides não atenderem as requisições.
    */
   public MaxPooling(int[] formEntrada, int[] formFiltro, int[] stride){
      this(formFiltro, stride);
      this.construir(formEntrada);
   }

   /**
    * Instancia uma camada MaxPooling.
    * <p>
    *    O formato de entrada da camada deve seguir o padrão:
    * </p>
    * <pre>
    *    formEntrada = (altura, largura, profundidade)
    * </pre>
    * @param entrada formato dos dados de entrada para a camada.
    */
   @Override
   public void construir(Object entrada){
      if(entrada == null){
         throw new IllegalArgumentException(
            "Formato de entrada fornecida para camada MaxPooling é nulo."
         );
      }
      if(entrada instanceof int[] == false){
         throw new IllegalArgumentException(
            "Objeto esperado para entrada da camada MaxPooling é do tipo int[], " +
            "objeto recebido é do tipo " + entrada.getClass().getTypeName()
         );
      }
      
      int[] e = (int[]) entrada;
      if(e.length != 3){
         throw new IllegalArgumentException(
            "O formato de entrada deve conter três elementos (altura, largura, profundidade)."
         );
      }
      this.formEntrada = new int[]{e[0], e[1], e[2]};

      this.formSaida = new int[3];
      formSaida[0] = (formEntrada[0] - formFiltro[0]) / this.stride[0] + 1;
      formSaida[1] = (formEntrada[1] - formFiltro[1]) / this.stride[1] + 1;
      formSaida[2] = formEntrada[2];
      
      this.entrada = new Mat[formEntrada[2]];
      this.gradEntrada = new Mat[formSaida[2]];
      this.saida = new Mat[formSaida[2]];

      for(int i = 0; i < formEntrada[2]; i++){
         this.entrada[i] = new Mat(this.formEntrada[0], this.formEntrada[1]);
         this.saida[i] = new Mat(this.formSaida[0], this.formSaida[1]);
         this.gradEntrada[i] = new Mat(this.formEntrada[0], this.formEntrada[1]);
      }
   }

   @Override
   public void inicializar(Inicializador iniKernel, Inicializador iniBias, double x){}

   @Override
   public void inicializar(Inicializador iniKernel, double x){}

   @Override
   public void calcularSaida(Object entrada){
      if(entrada instanceof Mat[]){
         Mat[] e = (Mat[]) entrada;
         utils.copiar(e, this.entrada);
         for(int i = 0; i < this.entrada.length; i++){
            aplicarMaxPooling(this.entrada[i], this.saida[i]);
         }
         
      }else if(entrada instanceof double[]){
         utils.copiar((double[]) entrada, this.entrada);
         for(int i = 0; i < this.entrada.length; i++){
            aplicarMaxPooling(this.entrada[i], this.saida[i]);
         }

      }else{
         throw new IllegalArgumentException(
            "Tipo de entrada \"" + entrada.getClass().getTypeName() + "\" não suportada."
         );
      }
   }

   private void aplicarMaxPooling(Mat entrada, Mat saida){
      for(int i = 0; i < saida.lin(); i++){
         for(int j = 0; j < saida.col(); j++){
            int linInicio = i * this.stride[0];
            int colIincio = j * this.stride[1];
            int linFim = Math.min(linInicio + this.formFiltro[0], entrada.lin());
            int colFim = Math.min(colIincio + this.formFiltro[1], entrada.col());
            double maxValor = Double.MIN_VALUE;
            
            for(int lin = linInicio; lin < linFim; lin++){
               for(int col = colIincio; col < colFim; col++){
                  double valor = entrada.elemento(lin, col);
                  if (valor > maxValor){
                     maxValor = valor;
                  }
               }
            }
            saida.editar(i, j, maxValor);
         }
      }
   }

   @Override
   public void calcularGradiente(Object gradSeguinte){
      if(gradSeguinte instanceof Mat[]){
         Mat[] grad = (Mat[]) gradSeguinte;   
         for(int i = 0; i < gradEntrada.length; i++){
            gradienteMaxPooling(this.entrada[i], grad[i], this.gradEntrada[i]);
         }
      
      }else{
         throw new IllegalArgumentException(
            "Formato de gradiente \" "+ gradSeguinte.getClass().getTypeName() +" \" não " +
            "suportado para camada de MaxPooling."
         );
      }
   }
   
   /**
    * Calcula e atualiza os gradientes da camada de Max Pooling em relação à entrada.
    * <p>
    *    Retroropaga os gradientes da camada seguinte para a camada de Max Pooling, considerando 
    *    a operação de agrupamento máximo. Ela calcula os gradientes em relação à entrada para as 
    *    camadas anteriores.
    * </p>
    * @param entrada entrada da camada.
    * @param gradSeguinte gradiente da camada seguinte.
    * @param gradEntrada gradiente de entrada da camada de max pooling.
    */
   private void gradienteMaxPooling(Mat entrada, Mat gradSeguinte, Mat gradEntrada){
      for(int i = 0; i < gradSeguinte.lin(); i++){
         for(int j = 0; j < gradSeguinte.col(); j++){
            int linInicio = i * this.stride[0];
            int colInicio = j * this.stride[1];
            int linFim = Math.min(linInicio + this.formFiltro[0], entrada.lin());
            int colFim = Math.min(colInicio + this.formFiltro[1], entrada.col());

            int[] posicaoMaximo = posicaoMaxima(entrada, linInicio, colInicio, linFim, colFim);
            int linMaximo = posicaoMaximo[0];
            int colMaximo = posicaoMaximo[1];

            double valorGradSeguinte = gradSeguinte.elemento(i, j);
            gradEntrada.editar(linMaximo, colMaximo, valorGradSeguinte);
         }
      }
   }
  
   /**
    * Encontra a posição do valor máximo em uma submatriz da matriz.
    * <p>
    *    Se houver múltiplos elementos com o valor máximo, a função retorna as coordenadas 
    *    do primeiro encontrado.
    * </p>
    * @param m matriz base.
    * @param linInicio índice inicial para linha.
    * @param colInicio índice final para a linha.
    * @param linFim índice inicial para coluna (exclusivo).
    * @param colFim índice final para coluna (exclusivo).
    * @return array representando as coordenadas (linha, coluna) do valor máximo
    * na submatriz.
    */
   private int[] posicaoMaxima(Mat m, int linInicio, int colInicio, int linFim, int colFim){
      int[] posMaximo = new int[]{linInicio, colInicio};
      double valMaximo = Double.NEGATIVE_INFINITY;
  
      for(int lin = linInicio; lin < linFim; lin++){
         for(int col = colInicio; col < colFim; col++){
            if(m.elemento(lin, col) > valMaximo){
               valMaximo = m.elemento(lin, col);
               posMaximo[0] = lin;
               posMaximo[1] = col;
            }
         }
      }
  
      return posMaximo;
   }
 
   @Override
   public int[] formatoEntrada(){
      return this.formEntrada;
   }

   @Override
   public int[] formatoSaida(){
      return this.formSaida;
   }

   @Override
   public int tamanhoSaida(){
      return this.saida.length * this.saida[0].lin() * this.saida[0].col();
   }

   /**
    * Retorna o formato do filtro usado pela camada.
    * @return dimensões do filtro de pooling.
    */
   public int[] formatoFiltro(){
      return this.formFiltro;
   }
      
   /**
    * Retorna o formato dos strides usado pela camada.
    * @return dimensões dos strides.
    */
   public int[] formatoStride(){
      return this.stride;
   }

   @Override
   public int numParametros(){
      return 0;
   }

   @Override
   public Mat[] saida(){
      return this.saida;
   }

   @Override
   public double[] saidaParaArray(){
      int id = 0;
      double[] saida = new double[this.tamanhoSaida()];

      for(int i = 0; i < this.saida.length; i++){
         double[] s = this.saida[i].paraArray();
         for(int j = 0; j < s.length; j++){
            saida[id++] = s[j];
         }
      }

      return saida;
   }

   @Override
   public Mat[] obterGradEntrada(){
      return this.gradEntrada;
   }
}
