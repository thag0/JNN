package rna.camadas;

import rna.core.Mat;
import rna.core.Utils;

/**
 * <h2>
 *    Camada flatten ou "achatadora"
 * </h2>
 * <p>
 *    Transforma os recebidos no formato sequencial.
 * </p>
 * <p>
 *    A camada de flatten não possui parâmetros treináveis nem função de ativação.
 * </p>
 */
public class Flatten extends Camada{

   /**
    * Utilitário.
    */
   Utils utils = new Utils();

   /**
    * Array contendo o formato de entrada da camada, de acordo com o formato:
    * <pre>
    * entrada = (altura, largura, profundidade)
    * </pre>
    */
   int[] formEntrada;

   /**
    * Array contendo o formato de saida da camada, de acordo com o formato:
    * <pre>
    * saida = (altura, largura)
    * </pre>
    */
   int[] formSaida;

   /**
    * Array de matrizes contendo os valores de entrada para a camada,
    * que serão usados para o processo de feedforward.
    * <p>
    *    O formato da entrada é dado por:
    * </p>
    * <pre>
    *entrada = [profundidade entrada]
    *entrada[n] = [alturaEntrada][larguraEntrada]
    * </pre>
    */
   public Mat[] entrada;
   
    /**
    * Array de matrizes contendo os valores dos gradientes usados para 
    * a retropropagação para camadas anteriores.
    */
   public Mat[] gradEntrada;

   /**
    * Matriz contendo a saída achatada da camada.
    * <p>
    *    Mesmo a saída sendo uma matriz, ela possui apenas
    *    uma linha e o número de colunas é equivalente a quantidade
    *    total de elementos da entrada.
    * </p>
    */
   public Mat saida;

   /**
    * Inicializa uma camada Flatten, que irá achatar a entrada recebida
    * no formato de um array unidimensional.
    * <p>
    *    É necessário construir a camada para que ela possa ser usada.
    * </p>
    */
   public Flatten(){}

   /**
    * Instancia uma camada Flatten, que irá achatar a entrada recebida
    * no formato de um array unidimensional.
    * <p>
    *    O formato de entrada da camada deve seguir o formato:
    * </p>
    * <pre>
    *    formEntrada = (altura, largura, profundidade)
    * </pre>
    * @param formEntrada formato dos dados de entrada para a camada.
    */
   public Flatten(int[] formEntrada){
      this.construir(formEntrada);
   }

   /**
    * Inicializa os parâmetros necessários para a camada Flatten.
    * <p>
    *    O formato de entrada deve ser um array contendo o tamanho de 
    *    cada dimensão e entrada da camada, e deve estar no formato:
    * </p>
    * <pre>
    *    entrada = (altura, largura, profundidade)
    * </pre>
    * Também pode ser aceito um objeto de entrada contendo apenas dois elementos,
    * eles serão formatados como:
    * <pre>
    *    entrada = (altura, largura)
    * </pre>
    * @param entrada formato de entrada para a camada.
    */
   @Override
   public void construir(Object entrada){
      if(entrada == null){
         throw new IllegalArgumentException(
            "Formato de entrada fornecida para camada Flatten é nulo."
         );
      }
      if(entrada instanceof int[] == false){
         throw new IllegalArgumentException(
            "Objeto esperado para entrada da camada Flatten é do tipo int[], " +
            "objeto recebido é do tipo " + entrada.getClass().getTypeName()
         );
      }

      int[] formatoEntrada = (int[]) entrada;
      int altura, largura, profundidade;
      if(formatoEntrada.length == 2){
         if(formatoEntrada[0] == 0 || formatoEntrada[1] == 0){
            throw new IllegalArgumentException(
               "Os valores recebidos para o formato de entrada devem ser maiores que zero, " +
               "recebido = [" + formatoEntrada[0] + ", " + formatoEntrada[1] + "]"
            );
         }

         altura = formatoEntrada[0];
         largura = formatoEntrada[1];
         profundidade = 1;
      
      }else if(formatoEntrada.length == 3){
         if(formatoEntrada[0] == 0 || formatoEntrada[1] == 0 || formatoEntrada[2] == 0){
            throw new IllegalArgumentException(
               "Os valores recebidos para o formato de entrada devem ser maiores que zero, " +
               "recebido = [" + formatoEntrada[0] + ", " + formatoEntrada[1] + ", " + formatoEntrada[2] + "]"
            );
         }
         altura = formatoEntrada[0];
         largura = formatoEntrada[1];
         profundidade = formatoEntrada[2];
      
      }else{
         throw new IllegalArgumentException(
            "O formato de entrada para a camada Flatten deve conter três " + 
            "elementos (altura, largura, profundidade) ou dois elementos (altura, largura), " +
            "objeto recebido possui " + formatoEntrada.length
         );
      }

      //inicialização de parâmetros

      this.formEntrada = new int[]{
         altura,
         largura,
         profundidade
      };

      int tamanho = 1;
      for(int i : this.formEntrada){
         tamanho *= i;
      }

      this.entrada = new Mat[formEntrada[2]];
      this.gradEntrada = new Mat[formEntrada[2]];
      for(int i = 0; i < this.entrada.length; i++){
         this.entrada[i] = new Mat(formEntrada[0], formEntrada[1]);
         this.gradEntrada[i] = new Mat(formEntrada[0], formEntrada[1]);
      }

      this.saida = new Mat(1, (this.entrada.length * this.entrada[0].lin() * this.entrada[0].col()));
      this.formSaida = new int[]{1, tamanho};

      this.construida = true;//camada pode ser usada.
   }

   @Override
   public void inicializar(double x){}

   /**
    * Achata os dados de entrada num formato sequencial.
    * <h3>
    *    Nota
    * </h3>
    * <p>
    *    Os dados de saída são armazenados numa matriz do tipo {@code Mat},
    *    que por sua vez possuem a quantidade de colunas equivalente a quantidade
    *    total de elementos de entrada.  
    * </p>
    * @param entrada dados de entrada que serão processados, objetos aceitos incluem:
    * {@code Mat[]}, {@code Mat} ou {@code double[]}.
    * @throws IllegalArgumentException caso a entrada fornecida não seja suportada 
    * pela camada.
    */
   @Override
   public void calcularSaida(Object entrada){
      super.verificarConstrucao();

      if(entrada instanceof Mat[]){
         Mat[] e = (Mat[]) entrada;

         for(int i = 0; i < this.entrada.length; i++){
            this.entrada[i].copiar(e[i]);
         }
      
      }else if(entrada instanceof Mat){
         this.entrada[0].copiar((Mat) entrada);

      }else if(entrada instanceof double[]){
         double[] e = (double[]) entrada;
         utils.copiar(e, this.entrada);
      
      }else{
         throw new IllegalArgumentException(
            "A camada Flatten não suporta entradas do tipo \"" + entrada.getClass().getTypeName() + "\"."
         );
      }

      int id = 0;
      for(int i = 0; i < this.entrada.length; i++){
         double[] arr = this.entrada[i].paraArray();
         for(int j = 0; j < arr.length; j++){
            this.saida.editar(0, id++, arr[j]);
         }
      }
   }

   /**
    * Desserializa os gradientes recebedos de volta para o mesmo formato de entrada.
    * @param gradSeguinte gradientes de entrada da camada seguinte.
    */
   @Override
   public void calcularGradiente(Object gradSeguinte){
      super.verificarConstrucao();

      if(gradSeguinte instanceof double[]){
         utils.copiar((double[]) gradSeguinte, this.gradEntrada);
      
      }else if(gradSeguinte instanceof Mat){
         Mat grads = (Mat) gradSeguinte;
         utils.copiar(grads.paraArray(), this.gradEntrada);
      
      }else{
         throw new IllegalArgumentException(
            "O gradiente seguinte para a camada Flatten deve ser do tipo \"double[]\" ou \"Mat\", " +
            "Objeto recebido é do tipo " + gradSeguinte.getClass().getTypeName()
         );
      }
   }

   @Override
   public Mat saida(){
      return this.saida;
   }

   @Override
   public double[] saidaParaArray(){
      super.verificarConstrucao();
      return this.saida.paraArray();
   }

   @Override
   public int tamanhoSaida(){
      super.verificarConstrucao();
      return this.saida.col();
   }

   @Override
   public int[] formatoEntrada(){
      super.verificarConstrucao();
      return this.formEntrada;
   }

   /**
    * Calcula o formato de saída da camada Flatten, que é disposto da
    * seguinte forma:
    * <pre>
    *    formato = (altura, largura)
    * </pre>
    * No caso da camada flatten, o formato também pode ser dito como:
    * <pre>
    *    formato = (1, elementosEntrada)
    * </pre>
    * Onde {@code elementosEntrada} é a quantidade total de elementos 
    * contidos no formato de entrada da camada.
    * @return formato de saída da camada
    */
    @Override
   public int[] formatoSaida(){
      super.verificarConstrucao();
      return new int[]{
         this.formSaida[0],
         this.formSaida[1]
      };
   }

   @Override
   public int numParametros(){
      return 0;
   }

   @Override
   public Object obterGradEntrada(){
      super.verificarConstrucao();
      return this.gradEntrada;
   }

}
