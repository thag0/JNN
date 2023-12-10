package rna.estrutura;

import rna.core.Mat;
import rna.inicializadores.Inicializador;

/**
 * 
 */
public class Flatten extends Camada{

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

   public Mat[] entrada;
   public Mat[] gradEntrada;
   public double[] saida;

   /**
    * Inicializa uma camada Flatten, que irá achatar a entrada recebida
    * no formato de um array unidimensional.
    * <p>
    *    É necessário construir a camada para que ela possa ser usada.
    * </p>
    */
   public Flatten(){

   }

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
      if(formatoEntrada.length != 3){
         throw new IllegalArgumentException(
            "O formato de entrada para a camada Flatten deve conter três " + 
            "elementos (altura, largura, profundidade), objeto recebido possui " + formatoEntrada.length
         );
      }
      if(formatoEntrada[0] == 0 || formatoEntrada[1] == 0 || formatoEntrada[2] == 0){
         throw new IllegalArgumentException(
            "Os valores recebidos para o formato de entrada devem ser maiores que zero, " +
            "recebido = [" + formatoEntrada[0] + ", " + formatoEntrada[1] + ", " + formatoEntrada[2] + "]"
         );
      }

      this.formEntrada = new int[]{
         formatoEntrada[0],
         formatoEntrada[1],
         formatoEntrada[2]
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

      this.saida = new double[tamanho];
      this.formSaida = new int[]{1, tamanho};

      this.construida = true;//camada pode ser usada.
   }

   @Override
   public void inicializar(Inicializador iniKernel, Inicializador iniBias, double x){
   }

   @Override
   public void inicializar(Inicializador iniKernel, double x){
   }

   @Override
   public void configurarId(int id){
      this.id = id;
   }

   @Override
   public void calcularSaida(Object entrada){
      if(entrada instanceof double[] == false){
         throw new IllegalArgumentException(
            "Os dados de entrada para a camada Flatten devem ser do tipo \"double[]\", " +
            "objeto recebido é do tipo \"" + entrada.getClass().getTypeName() + "\""
         );
      }

      double[] e = (double[]) entrada;
      System.arraycopy(e, 0, this.saida, 0, this.saida.length);
   }

   @Override
   public void calcularGradiente(Object gradSeguinte){
      if(gradSeguinte instanceof double[]){
         calcularGrad((double[]) gradSeguinte);
      
      }else if(gradSeguinte instanceof Mat){
         calcularGrad((Mat) gradSeguinte);
      
      }else{
         throw new IllegalArgumentException(
            "O gradiente seguinte para a camada Flatten deve ser do tipo \"double[]\" ou \"Mat\", " +
            "Objeto recebido é do tipo " + gradSeguinte.getClass().getTypeName()
         );
      }
   }

   private void calcularGrad(double[] grad){
      int id = 0, i, j, k;
      for(i = 0; i < this.gradEntrada.length; i++){
         for(j = 0; j < this.gradEntrada[i].lin; j++){
            for(k = 0; k < this.gradEntrada[i].col; k++){
               this.gradEntrada[i].editar(j, k, grad[id++]);
            }
         }
      }
   }
  
   private void calcularGrad(Mat grad){
      double[] g = grad.paraArray();
      calcularGrad(g);
   }

   @Override
   public double[] saidaParaArray(){
      return this.saida;
   }

   @Override
   public int tamanhoSaida(){
      return this.saida.length;
   }

   @Override
   public int[] formatoEntrada(){
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
      return new int[]{
         this.formSaida[0],
         this.formSaida[1]
      };
   }

   @Override
   public Object obterGradEntrada(){
      return this.gradEntrada;
   }

}
