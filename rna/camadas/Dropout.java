package rna.camadas;

import java.util.Random;

import rna.core.Tensor4D;
import rna.core.Utils;

/**
 * <h2>
 *    Camada de Abandono
 * </h2>
 * <p>
 *    O dropout (abandono) é uma técnica regularizadora usada para evitar
 *    overfitting para modelos muito grandes, melhorando na generalização.
 * </p>
 * <p>
 *    Durante o treinamento, o dropout "desativa" temporariamente aleatoriamente 
 *    algumas unidades/neurônios da camada, impedindo que eles contribuam para o 
 *    cálculo da saída. Isso força o modelo a aprender a partir de diferentes 
 *    subconjuntos dos dados em cada iteração, o que ajuda a evitar a dependência 
 *    excessiva de determinadas unidades e resulta numa representação mais generalista
 *    do conjunto de dados.
 * </p>
 * <p>
 *    A camada de dropout não possui parâmetros treináveis nem função de ativação.
 * </p>
 */
public class Dropout extends Camada implements Cloneable{

   /**
    * Taxa de abandono usada durante o treinamento.
    */
   private double taxa;

   /**
    * Formato de entrada da camada (profundidade, altura, largura).
    */
   int[] formEntrada;

   /**
    * Tensor contendo os valores de entrada para a camada.
    * <p>
    *    O formato da entrada é dado por:
    * </p>
    * <pre>
    *    entrada = (1, profundidade, altura, largura)
    * </pre>
    */
   public Tensor4D entrada;

   /**
    * Tensores contendo as máscaras que serão usadas durante
    * o processo de treinamento.
    * <p>
    *    O formato das máscaras é dado por:
    * </p>
    * <pre>
    *    mascara = (1, profundidade, altura, largura)
    * </pre>
    */
   public Tensor4D mascara;

   /**
    * Tensor contendo os valores de saída da camada.
    * <p>
    *    O formato de saída é dado por:
    * </p>
    * <pre>
    *    saida = (1, profundidade, altura, largura)
    * </pre>
    */
   public Tensor4D saida;

   /**
    * Tensor contendo os valores dos gradientes que
    * serão retropropagados durante o processo de treinamento.
    * <p>
    *    O formato dos gradientes é dado por:
    * </p>
    * <pre>
    *    gradEntrada = (1, profundidade, altura, largura)
    * </pre>
    */
   public Tensor4D gradEntrada;

   /**
    * Gerador de valores aleatórios.
    */
   Random random = new Random();//implementar configuração de seed

   /**
    * Utilitário.
    */
   Utils utils = new Utils();

   /**
    * Instancia uma nova camada de dropout, definindo a taxa
    * de abandono que será usada durante o processo de treinamento.
    * @param taxa taxa de dropout, um valor entre 0 e 1 representando a
    * taxa de abandono da camada.
    */
   public Dropout(double taxa){
      if(taxa <= 0 || taxa >= 1){
         throw new IllegalArgumentException(
            "\nO valor da taxa de dropout deve estar entre 0 e 1, " + 
            "recebido: " + taxa
         );
      }

      this.taxa = taxa;
   }

   /**
    * Instancia uma nova camada de dropout, definindo a taxa
    * de abandono que será usada durante o processo de treinamento.
    * @param taxa taxa de dropout, um valor entre 0 e 1 representando a
    * taxa de abandono da camada.
    * @param seed seed usada para o gerador de números aleatórios da camada.
    */
   public Dropout(double taxa, long seed){
      this(taxa);
      this.random.setSeed(seed);
   }

   @Override
   public void construir(Object entrada){
      if(entrada instanceof int[] == false){
         throw new IllegalArgumentException(
            "\nObjeto esperado para entrada da camada Dropout é do tipo int[], " +
            "objeto recebido é do tipo " + entrada.getClass().getTypeName()
         );
      }

      int[] formato = (int[]) entrada;
      if(utils.apenasMaiorZero(formato) == false){
         throw new IllegalArgumentException(
            "\nOs argumentos do formato de entrada devem ser maiores que zero."
         );
      }

      if(formato.length == 2){
         this.formEntrada = new int[]{
            1,
            1,
            formato[0],//altura
            formato[1],//largura
         };

      }else if(formato.length == 3){
         this.formEntrada = new int[]{
            1,
            formato[0],//profundidade
            formato[1],//altura
            formato[2],//largura
         };

      }else if(formato.length == 4){
         this.formEntrada = new int[]{
            1,
            formato[1],//profundidade
            formato[2],//altura
            formato[3],//largura
         };

      }else{
         throw new IllegalArgumentException(
            "\nA camada de dropout aceita formatos bidimensionais (altura, largura) " + 
            " tridimensionais (profundidade, altura, largura) " +
            " ou quadridimensionais (primeiro valor desconsiderado) " +
            "entrada recebida possui " + formato.length + " dimensões."
         );
      }
      
      this.entrada =     new Tensor4D(this.formEntrada);
      this.mascara =     new Tensor4D(this.entrada.shape());
      this.saida =       new Tensor4D(this.entrada.shape());
      this.gradEntrada = new Tensor4D(this.entrada.shape());

      setNomes();
      
      this.construida = true;//camada pode ser usada
   }

   @Override
   public void inicializar(){}

   /**
    * Configura uma seed fixa para geradores de números aleatórios da
    * camada.
    * @param seed nova seed.
    */
   public void setSeed(long seed){
      this.random.setSeed(seed);
   }

   @Override
   protected void setNomes(){
      entrada.nome("entrada");
      mascara.nome("máscara");
      saida.nome("saida");
      gradEntrada.nome("gradiente entrada");    
   }

   /**
    * <h2>
    *    Propagação direta através da camada Dropout
    * </h2>
    * <p>
    *    A máscara de dropout é gerada e utilizada apenas durante o processo
    *    de treinamento, durante isso, cada predição irá gerar uma máscara diferente,
    *    a máscara será usada como filtro para os dados de entrada e os valores de 
    *    unidades desativadas serão propagados como "0".
    * </p>
    * <p>
    *    Caso a camada não esteja {@code treinando}, por padrão a entrada é apenas
    *    repassada para a saída da camada.
    * </p>
    * <h3>
    *    Nota
    * </h3>
    * <p>
    *    Caso a entrada recebida seja um {@code Tensor4D}, os valores de entrada
    *    que serão considerados serão apenas o da primeira dimensão do tensor.
    * </p>
    * @param entrada dados de entrada para a camada, objetos aceitos são {@code Tensor4D}
    * ou {@code double[][][]}.
    */
   @Override
   public Tensor4D forward(Object entrada){
      verificarConstrucao();

      if(entrada instanceof Tensor4D){
         Tensor4D e = (Tensor4D) entrada;
         if(this.entrada.comparar3D(e) == false){
            throw new IllegalArgumentException(
               "\nDimensões de entrada " + e.shapeStr() + 
               "incompatível com as dimensões da entrada da camada " + this.entrada.shapeStr()
            );
         }

         this.entrada.copiar(e, 0);
   
      }else if(entrada instanceof double[][][]){
         double[][][] e = (double[][][]) entrada;
         this.entrada.copiar(e, 0);

      }else{
         throw new IllegalArgumentException(
            "\nEntrada aceita para a camada de Dropout deve ser do tipo " +
            this.entrada.getClass().getSimpleName() + " ou double[][][]" + 
            " , objeto recebido é do tipo \"" + entrada.getClass().getTypeName() + "\"."
         );
      }

      if(treinando){
         gerarMascaras();
         saida.copiar(this.entrada);
         saida.mult(mascara);

      }else{
         saida.copiar(this.entrada);
      }

      return saida.clone();
   }

   /**
    * Gera a máscara aleatória para cada camada de entrada que será 
    * usada durante o processo de treinamento.
    * <p>
    *    Exemplo:
    * </p>
    * <pre>
    *mascara = [
    *    1, 0, 0  
    *    0, 1, 1  
    *    0, 1, 0  
    *]
    * </pre>
    * Nos valores em que a máscara for igual a 1, o valor de entrada será
    * passado para a saída, nos valores iguais a 0, a entrada será desconsiderada. 
    */
   private void gerarMascaras(){
      mascara.map3D(0, (x) -> {
         return (random.nextDouble() >= taxa) ? (1 / (1-taxa)) : 0;
      });
   }

   /**
    * <h2>
    *    Propagação reversa através da camada Dropout
    * </h2>
    * Retropropaga os gradientes da camada suguinte, aplicando a máscara usada
    * no cálculo da saída.
    * <p>
    *    Caso a camada não esteja {@code treinando}, por padrão os gradientes são 
    *    apenas repassados para o gradiente de entrada da camada.
    * </p>
    * <h3>
    *    Nota
    * </h3>
    * <p>
    *    Serão considerados apenas os gradientes da primeira dimensão do tensor.
    * </p>
    * @param grad gradientes da camada seguiente.
    */
   @Override
   public Tensor4D backward(Object grad){
      verificarConstrucao();

      if(grad instanceof Tensor4D){
         Tensor4D g = (Tensor4D) grad;
         if(this.gradEntrada.comparar3D(g) == false){
            throw new IllegalArgumentException(
               "\nDimensões incompatíveis entre o gradiente recebido " + g.shapeStr() +
               "e o suportado pela camada " + gradEntrada.shapeStr()
            );
         }

         this.gradEntrada.copiar(g);

      }else{
         throw new IllegalArgumentException(
            "\nGradiente aceito para a camada de Dropout deve ser do tipo " + 
            this.gradEntrada.getClass().getTypeName() +
            " ,objeto recebido é do tipo \"" + grad.getClass().getTypeName() + "\"."
         );
      }

      if(treinando){
         gradEntrada.mult(mascara);
      }

      return gradEntrada.clone();
   }

   @Override
   public Tensor4D saida(){
      verificarConstrucao();
      return this.saida;
   }

   @Override
   public int[] formatoEntrada(){
      verificarConstrucao();
      return this.formEntrada;
   }

   @Override
   public int[] formatoSaida(){
      verificarConstrucao();
      return this.formEntrada;
   }

   @Override
   public int numParametros(){
      return 0;
   }

   /**
    * Retorna a taxa de dropout usada pela camada.
    * @return taxa de dropout da camada.
    */
   public double taxa(){
      return this.taxa;

   }

   @Override
   public Dropout clonar(){
      verificarConstrucao();

      try{
         Dropout clone = (Dropout) super.clone();
         clone.formEntrada = this.formEntrada.clone();
         clone.taxa = this.taxa;
         clone.random = new Random();

         clone.entrada = this.entrada.clone();
         clone.mascara = this.mascara.clone();
         clone.saida = this.saida.clone();
         clone.gradEntrada = this.gradEntrada.clone();

         return clone;
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }

   @Override
   public Tensor4D gradEntrada(){
      verificarConstrucao();
      return this.gradEntrada;
   }
}
