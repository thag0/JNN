package rna.otimizadores;

import rna.estrutura.Camada;

/**
 * Implementação do algoritmo de otimização Adam.
 * <p>
 *    O algoritmo ajusta os pesos da rede neural usando o gradiente descendente 
 *    com momento e a estimativa adaptativa de momentos de primeira e segunda ordem.
 * </p>
 * <p>
 * 	Os hiperparâmetros do Adam podem ser ajustados para controlar o 
 * 	comportamento do otimizador durante o treinamento.
 * </p>
 * <p>
 *    O Adam funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    var[i][j] -= (alfa * m[i][j]) / ((√ v[i][j]) + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code var} - variável que será otimizada (kernel, bias).
 * </p>
 * <p>
 *    {@code alfa} - correção aplicada a taxa de aprendizagem.
 * </p>
 * <p>
 *    {@code m} - coeficiente de momentum correspondente a variável que
 *    será otimizada;
 * </p>
 * <p>
 *    {@code v} - coeficiente de momentum de segunda orgem correspondente 
 *    a variável que será otimizada;
 * </p>
 * <p>
 *    {@code eps} - pequeno valor usado para evitar divisão por zero.
 * </p>
 * O valor de {@code alfa} é dado por:
 * <pre>
 * alfa = taxaAprendizagem * √(1- beta1ⁱ) / (1 - beta2ⁱ)
 * </pre>
 * Onde:
 * <p>
 *    {@code i} - contador de interações do Adam.
 * </p>
 * As atualizações de momentum de primeira e segunda ordem se dão por:
 *<pre>
 *m[i][j] += (1 - beta1) * (g  - m[i][j])
 *v[i][j] += (1 - beta2) * (g² - v[i][j])
 *</pre>
 * Onde:
 * <p>
 *    {@code beta1 e beta2} - valores de decaimento dos momentums de primeira
 *    e segunda ordem.
 * </p>
 * <p>
 *    {@code g} - gradiente correspondente a variável que será otimizada.
 * </p>
 */
public class Adam extends Otimizador{
   private static final double PADRAO_TA = 0.001;
   private static final double PADRAO_BETA1 = 0.9;
   private static final double PADRAO_BETA2 = 0.999;
   private static final double PADRAO_EPS = 1e-7; 

   /**
    * Valor de taxa de aprendizagem do otimizador.
    */
   private double taxaAprendizagem;

   /**
    * Decaimento do momentum.
    */
   private double beta1;
    
   /**
    * Decaimento do momentum de segunda ordem.
    */
   private double beta2;
    
   /**
    * Usado para evitar divisão por zero.
    */
   private double epsilon;

   /**
    * Coeficientes de momentum para os kernels.
    */
   private double[] m;

   /**
    * Coeficientes de momentum para os bias.
    */
   private double[] mb;

   /**
    * Coeficientes de momentum de segunda ordem para os kernels.
    */
   private double[] v;

   /**
    * Coeficientes de momentum de segunda ordem para os bias.
    */
   private double[] vb;
   
   /**
    * Contador de iterações.
    */
   long interacoes = 0;
 
   /**
    * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA taxa de aprendizagem do otimizador.
    * @param beta1 decaimento do momento de primeira ordem.
    * @param beta2 decaimento do momento de segunda ordem.
    * @param epsilon usado para evitar a divisão por zero.
    */
   public Adam(double tA, double beta1, double beta2, double epsilon){
      this.taxaAprendizagem = tA;
      this.beta1 = beta1;
      this.beta2 = beta2;
      this.epsilon = epsilon;
   }
 
   /**
    * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA taxa de aprendizagem do otimizador.
    */
   public Adam(double tA){
      this(tA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Adam </strong>.
    * <p>
    *    Os hiperparâmetros do Adam serão inicializados com os valores 
    *    padrão, que são:
    * </p>
    * <p>
    *    {@code taxaAprendizagem = 0.001}
    * </p>
    * <p>
    *    {@code beta1 = 0.9}
    * </p>
    * <p>
    *    {@code beta2 = 0.999}
    * </p>
    * <p>
    *    {@code epsilon = 1e-7}
    * </p>
    */
   public Adam(){
      this(PADRAO_TA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
   }

   @Override
   public void inicializar(Camada[] redec){
      int nKernel = 0;
      int nBias = 0;
      
      for(Camada camada : redec){
         if(camada.treinavel == false) continue;

         nKernel += camada.obterKernel().length;
         if(camada.temBias()){
            nBias += camada.obterBias().length;
         }         
      }

      this.m  = new double[nKernel];
      this.v  = new double[nKernel];
      this.mb = new double[nBias];
      this.vb = new double[nBias];
   }

   @Override
   public void atualizar(Camada[] redec){
      int idKernel = 0, idBias = 0;
      double g;

      interacoes++;
      double forcaB1 = Math.pow(beta1, interacoes);
      double forcaB2 = Math.pow(beta2, interacoes);
      double alfa = taxaAprendizagem * Math.sqrt(1 - forcaB2) / (1 - forcaB1);
   
      for(Camada camada : redec){
         if(camada.treinavel == false) continue;

         double[] kernel = camada.obterKernel();
         double[] gradK = camada.obterGradKernel();

         for(int i = 0; i < kernel.length; i++){
            g = gradK[i];
            m[idKernel] += (1 - beta1) * (g     - m[idKernel]);
            v[idKernel] += (1 - beta2) * ((g*g) - v[idKernel]);
            kernel[i] += (alfa * m[idKernel]) / (Math.sqrt(v[idKernel]) + epsilon);
            idKernel++;
         }
         camada.editarKernel(kernel);
         
         if(camada.temBias()){
            double[] bias = camada.obterBias();
            double[] gradB = camada.obterGradBias();

            for(int i = 0; i < bias.length; i++){
               g = gradB[i];
               mb[idBias] += (1 - beta1) * (g     - mb[idBias]);
               vb[idBias] += (1 - beta2) * ((g*g) - vb[idBias]);
               bias[i] += (alfa * mb[idBias]) / (Math.sqrt(vb[idBias]) + epsilon);
               idBias++;
            }
            camada.editarBias(bias);
         }     
      }
   }

   @Override
   public String info(){
      String buffer = "";

      String espacamento = "    ";
      buffer += espacamento + "TaxaAprendizagem: " + this.taxaAprendizagem + "\n";
      buffer += espacamento + "Beta1: " + this.beta1 + "\n";
      buffer += espacamento + "Beta2: " + this.beta2 + "\n";
      buffer += espacamento + "Epsilon: " + this.epsilon + "\n";

      return buffer;
   }

}
