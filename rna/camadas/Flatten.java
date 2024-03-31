package rna.camadas;

import rna.core.Tensor4D;
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
    *    entrada = (profundidade, altura, largura)
    * </pre>
    */
   int[] formEntrada;

   /**
    * Array contendo o formato de saida da camada, de acordo com o formato:
    * <pre>
    *    saida = (1, 1, elementosTotaisEntrada)
    * </pre>
    */
   int[] formSaida;

   /**
    * Tensor contendo os valores de entrada para a camada,
    * que serão usados para o processo de feedforward.
    * <p>
    *    O formato da entrada é dado por:
    * </p>
    * <pre>
    *    entrada = (1, profundidade, altura, largura)
    * </pre>
    */
   public Tensor4D entrada;
   
   /**
    * Tensor contendo os valores dos gradientes usados para 
    * a retropropagação para camadas anteriores.
    * <p>
    *    O formato dos gradientes é dado por:
    * </p>
    * <pre>
    *    gradEntrada = (1, profundidadeEntrada, alturaEntrada, larguraEntrada)
    * </pre>
    */
   public Tensor4D gradEntrada;

   /**
    * Tensor contendo a saída achatada da camada.
    * <p>
    *    Mesmo a saída sendo um tensor, ela possui apenas
    *    uma linha e o número de colunas é equivalente a quantidade
    *    total de elementos da entrada.
    * </p>
    */
   public Tensor4D saida;

   /**
    * Inicializa uma camada Flatten, que irá achatar a entrada recebida
    * no formato de um tensor unidimensional.
    * <p>
    *    É necessário construir a camada para que ela possa ser usada.
    * </p>
    */
   public Flatten(){}

   /**
    * Instancia uma camada Flatten, que irá achatar a entrada recebida
    * no formato de um tensor unidimensional.
    * <p>
    *    O formato de entrada da camada deve seguir o formato:
    * </p>
    * <pre>
    *    formEntrada = (profundidade, altura, largura)
    * </pre>
    * @param formEntrada formato dos dados de entrada para a camada.
    */
   public Flatten(int[] formEntrada){
      construir(formEntrada);
   }

   /**
    * Inicializa os parâmetros necessários para a camada Flatten.
    * <p>
    *    O formato de entrada deve ser um array contendo o tamanho de 
    *    cada dimensão de entrada da camada, e deve estar no formato:
    * </p>
    * <pre>
    *    entrada = (1, profundidade, altura, largura)
    * </pre>
    * Também pode ser aceito um objeto de entrada contendo apenas dois elementos,
    * eles serão formatados como:
    * <pre>
    *    entrada = (1, 1, altura, largura)
    * </pre>
    * @param entrada formato de entrada para a camada.
    */
   @Override
   public void construir(Object entrada){
      if(entrada == null){
         throw new IllegalArgumentException(
            "\nFormato de entrada fornecida para camada Flatten é nulo."
         );
      }
      if(entrada instanceof int[] == false){
         throw new IllegalArgumentException(
            "\nObjeto esperado para entrada da camada Flatten é do tipo int[], " +
            "objeto recebido é do tipo " + entrada.getClass().getTypeName()
         );
      }

      int[] formatoEntrada = (int[]) entrada;
      if(utils.apenasMaiorZero(formatoEntrada) == false){
         throw new IllegalArgumentException(
            "\nOs valores do formato de entrada devem ser maiores que zero."
         );
      }

      int profundidade, altura, largura;
      if(formatoEntrada.length == 4){
         profundidade = formatoEntrada[1];
         altura = formatoEntrada[2];
         largura = formatoEntrada[3];

      }else if(formatoEntrada.length == 3){
         profundidade = formatoEntrada[0];
         altura = formatoEntrada[1];
         largura = formatoEntrada[2];
      
      }else if(formatoEntrada.length == 2){
         profundidade = 1;
         altura = formatoEntrada[0];
         largura = formatoEntrada[1];
      
      }else{
         throw new IllegalArgumentException(
            "O formato de entrada para a camada Flatten deve conter dois " + 
            "elementos (altura, largura), três elementos (profundidade, altura, largura), " +
            " ou quatro elementos (primeiro desconsiderado) " +
            "objeto recebido possui " + formatoEntrada.length + " elementos."
         );
      }

      //inicialização de parâmetros

      this.formEntrada = new int[]{
         1,
         profundidade,
         altura,
         largura
      };

      int tamanho = 1;
      for(int i : this.formEntrada){
         tamanho *= i;
      }

      this.formSaida = new int[]{1, 1, 1, tamanho};

      this.entrada = new Tensor4D(formEntrada);
      this.gradEntrada = new Tensor4D(this.entrada.shape());
      this.saida = new Tensor4D(formSaida);

      setNomes();

      this.construida = true;//camada pode ser usada.
   }

   @Override
   public void inicializar(){}

   @Override
   public void setSeed(long seed){}

   @Override
   protected void setNomes(){
      this.entrada.nome("entrada");
      this.saida.nome("saída");
      this.gradEntrada.nome("gradiente entrada");     
   }

   /**
    * <h2>
    *    Propagação direta através da camada Flatten
    * </h2>
    * Achata os dados de entrada num formato sequencial.
    * @param entrada dados de entrada que serão processados, objetos aceitos incluem:
    * {@code Tensor4D}, {@code double[][][]} ou {@code double[]}.
    * @throws IllegalArgumentException caso a entrada fornecida não seja suportada 
    * pela camada.
    */
   @Override
   public Tensor4D forward(Object entrada){
      verificarConstrucao();

      if(entrada instanceof Tensor4D){
         Tensor4D e = (Tensor4D) entrada;
         if(this.entrada.comparar3D(e) == false){
            throw new IllegalArgumentException(
               "\nDimensões da entrada recebida " + e.shapeStr() +
               " incompatíveis com a entrada da camada " + this.entrada.shapeStr()
            );
         }

         this.entrada.copiar(e.array3D(0), 0);

      }else if(entrada instanceof double[][][]){
         double[][][] e = (double[][][]) entrada;
         this.entrada.copiar(e, 0);

      }else if(entrada instanceof double[]){
         double[] e = (double[]) entrada;
         this.entrada.copiarElementos(e);
      
      }else{
         throw new IllegalArgumentException(
            "A camada Flatten não suporta entradas do tipo \"" + entrada.getClass().getTypeName() + "\"."
         );
      }

      saida.copiarElementos(this.entrada.paraArray());

      return saida.clone();
   }

   /**
    * <h2>
    *    Propagação reversa através da camada Flatten
    * </h2>
    * Desserializa os gradientes recebedos de volta para o mesmo formato de entrada.
    * @param grad gradientes de entrada da camada seguinte, objetos aceitos incluem:
    * {@code Tensor4D} ou {@code double[]}.
    */
   @Override
   public Tensor4D backward(Object grad){
      verificarConstrucao();

      if(grad instanceof Tensor4D){
         Tensor4D g = (Tensor4D) grad;
         if(g.tamanho() != this.gradEntrada.tamanho()){
            throw new IllegalArgumentException(
               "\nDimensões do gradiente recebido " + g.shapeStr() +
               "inconpatíveis com o suportado pela camada " + this.gradEntrada.shapeStr()
            );
         }

         this.gradEntrada.copiarElementos(g.paraArray());
      
      }else if(grad instanceof double[]){
         double[] g = (double[]) grad;
         gradEntrada.copiarElementos(g);
      
      }else{
         throw new IllegalArgumentException(
            "O gradiente seguinte para a camada Flatten deve ser do tipo \"double[]\" ou \"Tensor4D\", " +
            "Objeto recebido é do tipo " + grad.getClass().getTypeName()
         );
      }

      return gradEntrada.clone();
   }
   
   @Override
   public Tensor4D saida(){
      verificarConstrucao();
      return this.saida;
   }

   @Override
   public double[] saidaParaArray(){
      verificarConstrucao();
      return this.saida.paraArray();
   }

   @Override
   public int tamanhoSaida(){
      verificarConstrucao();
      return this.saida.tamanho();
   }

   @Override
   public int[] formatoEntrada(){
      verificarConstrucao();
      return this.formEntrada;
   }

   /**
    * Calcula o formato de saída da camada Flatten, que é disposto da
    * seguinte forma:
    * <pre>
    *    formato = (1, 1, 1, elementosEntrada)
    * </pre>
    * Onde {@code elementosEntrada} é a quantidade total de elementos 
    * contidos no formato de entrada da camada.
    * @return formato de saída da camada
    */
    @Override
   public int[] formatoSaida(){
      verificarConstrucao();
      
      return new int[]{
         1,
         1,
         1,
         tamanhoSaida()
      };
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
