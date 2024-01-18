package rna.camadas;

import java.util.Random;

import rna.core.Mat;
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
    * Formato de entrada da camada (altura, largura, profundidade).
    */
   int[] formEntrada;

   /**
    * Array de matrizes contendo os valores de entrada para a camada.
    * <p>
    *    O formato da entrada é dado por:
    * </p>
    * <pre>
    *entrada = [profundidade]
    *entrada[n] = [altura][largura]
    * </pre>
    */
   public Mat[] entrada;

   /**
    * Array de matrizes contendo as máscaras que serão usadas durante
    * o processo de treinamento.
    * <p>
    *    O formato das máscaras é dado por:
    * </p>
    * <pre>
    *mascara = [profundidade]
    *mascara[n] = [altura][largura]
    * </pre>
    */
   public Mat[] mascara;

   /**
    * Array de matrizes contendo os valores de saída da camada.
    * <p>
    *    O formato de saída é dado por:
    * </p>
    * <pre>
    *saida = [profundidade]
    *saida[n] = [altura][largura]
    * </pre>
    */
   public Mat[] saida;

   /**
    * Array de matrizes contendo os valores dos gradientes que
    * serão retropropagados durante o processo de treinamento.
    * <p>
    *    O formato dos gradientes é dado por:
    * </p>
    * <pre>
    *grad = [profundidade]
    *grad[n] = [altura][largura]
    * </pre>
    */
   public Mat[] gradEntrada;

   /**
    * Gerador de valores aleatórios.
    */
   Random random = new Random();//talvez implementar configuração de seed

   /**
    * 
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
            "O valor da taxa de dropout deve estar entre 0 e 1, " + 
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
            "Objeto esperado para entrada da camada Dropout é do tipo int[], " +
            "objeto recebido é do tipo " + entrada.getClass().getTypeName()
         );
      }

      int[] formato = (int[]) entrada;
      if(utils.apenasMaiorZero(formato) == false){
         throw new IllegalArgumentException(
            "Os argumentos do formato de entrada devem ser maiores que zero."
         );
      }

      if(formato.length == 2){
         this.formEntrada = new int[]{
            formato[0],
            formato[1],
            1
         };

      }else if(formato.length == 3){
         this.formEntrada = new int[]{
            formato[0],
            formato[1],
            formato[2]
         };

      }else{
         throw new IllegalArgumentException(
            "A camada de dropout aceita formatos bidimensionais (altura, largura) " + 
            " ou tridimensionais (altura, largura, profundidade), " +
            "entrada recebida possui " + formato.length + " dimensões."
         );
      }

      this.entrada =     new Mat[this.formEntrada[2]];
      this.mascara =     new Mat[this.formEntrada[2]];
      this.saida =       new Mat[this.formEntrada[2]];
      this.gradEntrada = new Mat[this.formEntrada[2]];
      
      for(int i = 0; i < this.formEntrada[2]; i++){
         this.entrada[i] =     new Mat(this.formEntrada[0], this.formEntrada[1]);
         this.mascara[i] =     new Mat(this.formEntrada[0], this.formEntrada[1]);
         this.saida[i] =       new Mat(this.formEntrada[0], this.formEntrada[1]);
         this.gradEntrada[i] = new Mat(this.formEntrada[0], this.formEntrada[1]);
      }
      this.construida = true;
   }

   @Override
   public void inicializar(double x){}

   /**
    * Propaga os dados de entrada recebido pela camada de dropout.
    * <p>
    *    A máscara de dropout é gerada e utilizada apenas durante o processo
    *    de treinamento, durante isso, cada predição irá gerar uma máscara diferente,
    *    a máscara será usada como filtro para os dados de entrada e os valores de 
    *    unidades desativadas serão propagados como "0".
    * </p>
    */
   @Override
   public void calcularSaida(Object entrada){
      super.verificarConstrucao();

      if(entrada instanceof Mat){
         this.entrada[0].copiar((Mat) entrada);

      }else if(entrada instanceof Mat[]){
         Mat[] e = (Mat[]) entrada;
         for(int i = 0; i < this.formEntrada[2]; i++){
            this.entrada[i].copiar(e[i]);
         }

      }else{
         throw new IllegalArgumentException(
            "Entrada aceita para a camada de Dropout deve ser do tipo Mat ou Mat[], " + 
            "objeto recebido é do tipo \"" + entrada.getClass().getTypeName() + "\"."
         );
      }

      if(this.treinando){
         gerarMascaras();
         int lin = this.saida[0].lin(), col = this.saida[0].col();
         double res;
         for(int i = 0; i < this.formEntrada[2]; i++){
            for(int j = 0; j < lin; j++){
               for(int k = 0; k < col; k++){
                  res = this.mascara[i].elemento(j, k) * this.entrada[i].elemento(j, k);
                  this.saida[i].editar(j, k, res);
               }
            }
         }

      }else{
         for(int i = 0; i < this.formEntrada[2]; i++){
            this.saida[i].copiar(this.entrada[i]);
         }
      }
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
    */
   private void gerarMascaras(){
      for(int i = 0; i < this.formEntrada[2]; i++){
         this.mascara[i].mapear((x) -> {
            return (this.random.nextDouble() > this.taxa) ? 1 : 0;
         });
      }
   }

   /**
    * Retropropaga os gradientes da camada suguinte, aplicando a máscara usada
    * no cálculo da saída.
    * @param gradSeguinte gradientes da camada seguiente.
    */
   @Override
   public void calcularGradiente(Object gradSeguinte){
      super.verificarConstrucao();

      if(gradSeguinte instanceof Mat){
         this.gradEntrada[0].copiar((Mat) gradSeguinte);

      }else if(gradSeguinte instanceof Mat[]){
         Mat[] g = (Mat[]) gradSeguinte;
         for(int i = 0; i < this.formEntrada[2]; i++){
            this.gradEntrada[i].copiar(g[i]);
         }

      }else{
         throw new IllegalArgumentException(
            "Gradiente aceito para a camada de Dropout deve ser do tipo Mat ou Mat[], " + 
            "objeto recebido é do tipo \"" + entrada.getClass().getTypeName() + "\"."
         );
      }

      for(int i = 0; i < this.formEntrada[2]; i++){
         this.gradEntrada[i].mult(this.mascara[i]);
      }
   }

   @Override
   public Object saida(){
      super.verificarConstrucao();
      return this.saida;
   }

   @Override
   public int[] formatoEntrada(){
      super.verificarConstrucao();
      return this.formEntrada;
   }

   @Override
   public int[] formatoSaida(){
      super.verificarConstrucao();
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
      try{
         Dropout clone = (Dropout) super.clone();
         clone.formEntrada = this.formEntrada.clone();
         clone.taxa = this.taxa;
         clone.random = new Random();

         int profundidade = this.formEntrada[2];
         clone.entrada = new Mat[profundidade];
         clone.mascara = new Mat[profundidade];
         clone.saida = new Mat[profundidade];
         clone.gradEntrada = new Mat[profundidade];
         
         for(int i = 0; i < profundidade; i++){
            clone.entrada[i] = this.entrada[i].clone();
            clone.mascara[i] = this.mascara[i].clone();
            clone.saida[i] = this.saida[i].clone();
            clone.gradEntrada[i] = this.gradEntrada[i].clone();
         }

         return clone;
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }

   @Override
   public Mat[] obterGradEntrada(){
      super.verificarConstrucao();
      return this.gradEntrada;
   }
}
