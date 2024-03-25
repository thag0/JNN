package rna.camadas;

import rna.core.Tensor4D;
import rna.core.Utils;

/**
 * <h2>
 *    Camada de agrupamento médio
 * </h2>
 * <p>
 *    A camada de agrupamento máximo é um componente utilizado para reduzir a 
 *    dimensionalidade espacial dos dados, preservando as características mais 
 *    importantes para a saída.
 * </p>
 * <p>
 *    Durante a operação de agrupamento médio, a entrada é dividida em regiões 
 *    menores usando uma máscara e a média de cada região é calculada e salva. 
 *    Essencialmente, a camada realiza a operação de subamostragem, calculando a 
 *    média das informações em cada região.
 * </p>
 * Exemplo simples de operação Avg Pooling para uma região 2x2 com máscara 2x2:
 * <pre>
 *entrada = [
 *    1, 2
 *    3, 4
 *]
 * 
 *saida = 2.5
 * </pre>
 * <p>
 *    A camada de avg pooling não possui parâmetros treináveis nem função de ativação.
 * </p>
 */
public class AvgPooling extends Camada{

   /**
    * Utilitario.
    */
   Utils utils = new Utils();

   /**
    * Dimensões dos dados de entrada (1, profundidade, altura, largura)
    */
   private int[] formEntrada;

   /**
    * Dimensões dos dados de saída (1, profundidade, altura, largura)
    */
   private int[] formSaida;

   /**
    * Tensor contendo os dados de entrada da camada.
    * <p>
    *    O formato da entrada é dado por:
    * </p>
    * <pre>
    *    entrada = (1, profundidade, altura, largura)
    * </pre>
    */
   public Tensor4D entrada;

   /**
    * Tensor contendo os dados de saída da camada.
    * <p>
    *    O formato de entrada varia dependendo da configuração da
    *    camada (filtro, strides) mas é dado como:
    * </p>
    * <pre>
    *largura = (larguraEntrada = larguraFiltro) / larguraStride + 1;
    *altura = (alturaEntrada = alturaFiltro) / alturaStride + 1;
    * </pre>
    * <p>
    *    Com isso o formato de saída é dado por:
    * </p>
    * <pre>
    *    saida = (1, profundidade, altura, largura)
    * </pre>
    * Essa relação é válida pra cada canal de entrada.
    */
   public Tensor4D saida;

   /**
    * Tensor contendo os gradientes que serão
    * retropropagados para as camadas anteriores.
    * <p>
    *    O formato do gradiente de entrada é dado por:
    * </p>
    * <pre>
    *    entrada = (1, profundidadeEntrada, alturaEntrada, larguraEntrad)
    * </pre>
    */
   public Tensor4D gradEntrada;

   /**
    * Formato do filtro de pooling (altura, largura).
    */
   private int[] formFiltro;

   /**
    * Valores de stride (altura, largura).
    */
   private int[] stride;

   /**
    * Instancia uma nova camada de average pooling, definindo o formato do
    * filtro que será aplicado em cada entrada da camada.
    * <p>
    *    O formato do filtro deve conter as dimensões da entrada da
    *    camada (altura, largura).
    * </p>
    * <p>
    *    Por padrão, os valores de strides serão os mesmos usados para
    *    as dimensões do filtro, exemplo:
    * </p>
    * <pre>
    *filtro = (2, 2)
    *stride = (2, 2) // valor padrão
    * </pre>
    * @param formFiltro formato do filtro de average pooling.
    * @throws IllegalArgumentException se o formato do filtro não atender as
    * requisições.
    */
   public AvgPooling(int[] formFiltro){
      if(formFiltro == null){
         throw new IllegalArgumentException(
            "\nO formato do filtro não pode ser nulo."
         );
      }
      if(formFiltro.length != 2){
         throw new IllegalArgumentException(
            "\nO formato do filtro deve conter dois elementos (altura, largura)."
         );
      }
      if(utils.apenasMaiorZero(formFiltro) == false){
         throw new IllegalArgumentException(
            "\nOs valores de dimensões do filtro devem ser maiores que zero."
         );
      }

      this.formFiltro = formFiltro;
      this.stride = new int[]{
         formFiltro[0],
         formFiltro[1]
      };
   }

   /**
    * Instancia uma nova camada de average pooling, definindo o formato do filtro 
    * e os strides (passos) que serão aplicados em cada entrada da camada.
    * <p>
    *    O formato do filtro e dos strides devem conter as dimensões da entrada 
    *    da camada (altura, largura).
    * </p>
    * @param formFiltro formato do filtro de average pooling.
    * @param stride strides que serão aplicados ao filtro.
    * @throws IllegalArgumentException se o formato do filtro não atender as
    * requisições.
    * @throws IllegalArgumentException se os strides não atenderem as requisições.
    */
   public AvgPooling(int[] formFiltro, int[] stride){
      if(formFiltro == null){
         throw new IllegalArgumentException(
            "\nO formato do filtro não pode ser nulo."
         );
      }
      if(formFiltro.length != 2){
         throw new IllegalArgumentException(
            "\nO formato do filtro deve conter três elementos (altura, largura)."
         );
      }
      if(utils.apenasMaiorZero(formFiltro) == false){
         throw new IllegalArgumentException(
            "\nOs valores de dimensões do filtro devem ser maiores que zero."
         );
      }

      if(stride == null){
         throw new IllegalArgumentException(
            "\nO formato do filtro não pode ser nulo."
         );
      }
      if(stride.length != 2){
         throw new IllegalArgumentException(
            "\nO formato para os strides deve conter dois elementos (altura, largura)."
         );
      }
      if(utils.apenasMaiorZero(stride) == false){
         throw new IllegalArgumentException(
            "\nOs valores para os strides devem ser maiores que zero."
         );
      }

      this.formFiltro = formFiltro;
      this.stride = stride;
   }

   /**
    * Instancia uma nova camada de average pooling, definindo o formato do filtro, 
    * formato de entrada e os strides (passos) que serão aplicados em cada entrada 
    * da camada.
    * <p>
    *    O formato do filtro e dos strides devem conter as dimensões da entrada 
    *    da camada (altura, largura).
    * </p>
    * A camada será automaticamente construída usando o formato de entrada especificado.
    * @param formEntrada formato de entrada para a camada.
    * @param formFiltro formato do filtro de average pooling.
    * @param stride strides que serão aplicados ao filtro.
    * @throws IllegalArgumentException se o formato do filtro não atender as
    * requisições.
    * @throws IllegalArgumentException se os strides não atenderem as requisições.
    */
   public AvgPooling(int[] formEntrada, int[] formFiltro, int[] stride){
      this(formFiltro, stride);
      construir(formEntrada);
   }

   /**
    * Constroi a camada AvgPooling, inicializando seus atributos.
    * <p>
    *    O formato de entrada da camada deve seguir o padrão:
    * </p>
    * <pre>
    *    formEntrada = (profundidade, altura, largura)
    * </pre>
    * <h3>
    *    Nota
    * </h3>
    * <p>
    *    Caso o formato de entrada contenha quatro elementos, o primeiro
    *    valor é descondiderado.
    * </p>
    * @param entrada formato dos dados de entrada para a camada.
    */
   @Override
   public void construir(Object entrada){
      if(entrada == null){
         throw new IllegalArgumentException(
            "\nFormato de entrada fornecida para camada " + nome() + " é nulo."
         );
      }
      if(entrada instanceof int[] == false){
         throw new IllegalArgumentException(
            "\nObjeto esperado para entrada da camada " + nome() + " é do tipo int[], " +
            "objeto recebido é do tipo " + entrada.getClass().getTypeName()
         );
      }
      
      int[] e = (int[]) entrada;
      if(e.length == 4){
         this.formEntrada = new int[]{1, e[1], e[2], e[3]};
      
      }else if(e.length == 3){
         this.formEntrada = new int[]{1, e[0], e[1], e[2]};
      }else{         
         throw new IllegalArgumentException(
            "\nO formato de entrada deve conter três elementos (profundidade, altura, largura) ou " +
            "quatro elementos (primeiro elementos desconsiderado)" +
            "formato recebido possui " + e.length + " elementos."
         );
      }

      this.formSaida = new int[4];
      formSaida[0] = 1;
      formSaida[1] = formEntrada[1];//profundidade
      formSaida[2] = (formEntrada[2] - formFiltro[0]) / this.stride[0] + 1;//altura
      formSaida[3] = (formEntrada[3] - formFiltro[1]) / this.stride[1] + 1;//largura
      
      this.entrada = new Tensor4D(formEntrada);
      this.gradEntrada = new Tensor4D(this.entrada);
      this.saida = new Tensor4D(formSaida);

      setNomes();

      this.construida = true;//camada pode ser usada
   }

   @Override
   public void inicializar(){}

   @Override
   protected void setNomes(){
      this.entrada.nome("Entrada");
      this.gradEntrada.nome("Gradiente entrada");
      this.saida.nome("Saída");
   }

   @Override
   public void calcularSaida(Object entrada){
      verificarConstrucao();

      if(entrada instanceof Tensor4D){
         Tensor4D e = (Tensor4D) entrada;

         if(this.entrada.comparar3D(e) == false){
            throw new IllegalArgumentException(
               "\nDimensões da entrada recebida " + e.shapeStr() +
               " incompatíveis com a entrada da camada " + this.entrada.shapeStr()
            );
         }

         this.entrada.copiar(e);
         
      }else if(entrada instanceof double[][][]){
         double[][][] e = (double[][][]) entrada;
         this.entrada.copiar(e, 0);

      }else{
         throw new IllegalArgumentException(
            "\nTipo de entrada \"" + entrada.getClass().getTypeName() + "\" não suportada."
         );
      }

      int profundidade = formEntrada[1];
      for(int i = 0; i < profundidade; i++){
         aplicar(this.entrada, this.saida, i);
      }
   }

   /**
    * Calcula a média dos valores encontrados na entrada de acordo com as
    * configurações de filtro e strides.
    * @param entrada tensor de entrada.
    * @param saida tensor de destino.
    * @param prof índice de profundidade da operação.
    */
   private void aplicar(Tensor4D entrada, Tensor4D saida, int prof){
      int alturaEntrada = entrada.dim3();
      int larguraEntrada = entrada.dim4();
      int alturaSaida = saida.dim3();
      int larguraSaida = saida.dim4();
  
      for(int i = 0; i < alturaSaida; i++){
         for(int j = 0; j < larguraSaida; j++){
            int linInicio = i * this.stride[0];
            int colInicio = j * this.stride[1];
            int linFim = Math.min(linInicio + this.formFiltro[0], alturaEntrada);
            int colFim = Math.min(colInicio + this.formFiltro[1], larguraEntrada);
            double soma = 0;
            int cont = 0;

            for(int lin = linInicio; lin < linFim; lin++){
               for(int col = colInicio; col < colFim; col++){
                  soma += entrada.get(0, prof, lin, col);
                  cont++;
               }
            }

            saida.set((soma/cont), 0, prof, i, j);
         }
      }
   }

   @Override
   public void calcularGradiente(Object gradSeguinte){
       verificarConstrucao();
   
       if(gradSeguinte instanceof Tensor4D){
           Tensor4D g = (Tensor4D) gradSeguinte;
           int profundidade = formEntrada[1];   
           for(int i = 0; i < profundidade; i++){
               gradAvgPool(this.entrada, g, this.gradEntrada, i);
           }
       
       }else{
           throw new IllegalArgumentException(
               "Formato de gradiente \" "+ gradSeguinte.getClass().getTypeName() +" \" não " +
               "suportado para camada de AvgPooling."
           );
       }
   }

   /**
    * Calcula e atualiza os gradientes da camada de Avg Pooling em relação à entrada.
    * <p>
    *    Retroropaga os gradientes da camada seguinte para a camada de Avg Pooling, considerando 
    *    a operação de agrupamento médio. Ela calcula os gradientes em relação à entrada para as 
    *    camadas anteriores.
    * </p>
    * @param entrada entrada da camada.
    * @param gradSeguinte gradiente da camada seguinte.
    * @param gradEntrada gradiente de entrada da camada de Avg pooling.
    * @param prof índice de profundidade da operação.
    */
   private void gradAvgPool(Tensor4D entrada, Tensor4D gradSeguinte, Tensor4D gradEntrada, int prof){
      int alturaEntrada = entrada.dim3();
      int larguraEntrada = entrada.dim4();
      int alturaGradSeguinte = gradSeguinte.dim3();
      int larguraGradSeguinte = gradSeguinte.dim4();

      for(int i = 0; i < alturaGradSeguinte; i++){
         for(int j = 0; j < larguraGradSeguinte; j++){
            int linInicio = i * this.stride[0];
            int colInicio = j * this.stride[1];
            int linFim = Math.min(linInicio + formFiltro[0], alturaEntrada);
            int colFim = Math.min(colInicio + formFiltro[1], larguraEntrada);

            double grad = gradSeguinte.get(0, prof, i, j);
            double mediaGrad = grad / (formFiltro[0] * formFiltro[1]);

            for(int lin = linInicio; lin < linFim; lin++){
               for(int col = colInicio; col < colFim; col++){
                  gradEntrada.set(mediaGrad, 0, prof, lin, col);
               }
            }
         }
      }
   }

   @Override
   public Tensor4D saida(){
      verificarConstrucao();
      return this.saida;
   }

   @Override
   public int[] formatoSaida(){
      verificarConstrucao();
      return this.formSaida;
   }

   @Override
   public int[] formatoEntrada(){
      verificarConstrucao();
      return this.formEntrada;
   }

   /**
    * Retorna o formato do filtro (altura, largura) usado pela camada.
    * @return formato do filtro da camada.
    */
   public int[] formatoFiltro(){
      verificarConstrucao();
      return this.formFiltro;
   }

   /**
    * Retorna o formato dos strides (altura, largura) usado pela camada.
    * @return formato dos strides da camada.
    */
   public int[] formatoStride(){
      verificarConstrucao();
      return this.stride;
   }

   @Override
   public int numParametros(){
      return 0;
   }

   @Override
   public Tensor4D gradEntrada(){
      verificarConstrucao();
      return this.gradEntrada;
   }
   
}
