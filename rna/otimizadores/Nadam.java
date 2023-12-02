package rna.otimizadores;

import rna.core.Mat;
import rna.estrutura.Densa;

/**
 * Implementação do algoritmo de otimização Nadam.
 * <p>
 *    O algoritmo ajusta os pesos da rede neural usando o gradiente descendente 
 *    com momento e a estimativa adaptativa de momentos de primeira e segunda ordem.
 * </p>
 * O adicional do Nadam é usar o acelerador de nesterov durante a correção dos
 * pesos da rede.
 * <p>
 *    O Nadam funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v[i][j] -= (tA * mc) / ((√ vc) + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - valor de taxa de aprendizagem do otimizador.
 * </p>
 * <p>
 *    {@code mc} - valor de momentum corrigido
 * </p>
 * <p>
 *    {@code m2c} - valor de velocidade (momentum de segunda ordem) corrigido
 * </p>
 * Os valores de momentum e velocidade corrigidos se dão por:
 * <pre>
 *mc = ((beta1 * m[[i][j]) + ((1 - beta1) * g[i][j])) / (1 - beta1ⁱ)
 *vc = (beta2 * v[i][j]) / (1 - beta2ⁱ)
 * </pre>
 * Onde:
 * <p>
 *    {@code m} - valor de momentum correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code v} - valor de velocidade correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code g} - gradiente correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code i} - contador de interações do otimizador.
 * </p>
 */
public class Nadam extends Otimizador{
   private static final double PADRAO_TA = 0.001;
   private static final double PADRAO_BETA1 = 0.9;
   private static final double PADRAO_BETA2 = 0.999;
   private static final double PADRAO_EPS = 1e-7;

   /**
    * Valor de taxa de aprendizagem do otimizador.
    */
   private double taxaAprendizagem;

   /**
    * Usado para evitar divisão por zero.
    */
   private double epsilon;

   /**
    * decaimento do momentum.
    */
   private double beta1;

   /**
    * decaimento do momentum de segunda ordem.
    */
   private double beta2;

   /**
    * Coeficientes de momentum.
    */
   private Mat[] m;
   
   /**
    * Coeficientes de momentum.
    */
   private Mat[] mb;

   /**
    * Coeficientes de momentum de segunda orgem.
    */
   private Mat[] v;

   /**
    * Coeficientes de momentum de segunda orgem.
    */
   private Mat[] vb;

   /**
    * Contador de iterações.
    */
   long interacoes = 0;

   /**
    * Inicializa uma nova instância de otimizador <strong> Nadam </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
    * @param beta1 decaimento do momento de primeira ordem.
    * @param beta2 decaimento da segunda ordem.
    * @param epsilon usado para evitar a divisão por zero.
    */
   public Nadam(double tA, double beta1, double beta2, double epsilon){
      this.taxaAprendizagem = tA;
      this.beta1 = beta1;
      this.beta2 = beta2;
      this.epsilon = epsilon;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Nadam </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
    */
   public Nadam(double tA){
      this(tA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Nadam </strong>.
    * <p>
    *    Os hiperparâmetros do Nadam serão inicializados com os valores padrão.
    * </p>
    */
   public Nadam(){
      this(PADRAO_TA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
   }

   @Override
   public void inicializar(Densa[] redec){
      this.m  = new Mat[redec.length];
      this.v  = new Mat[redec.length];
      this.mb = new Mat[redec.length];
      this.vb = new Mat[redec.length];
   
      for(int i = 0; i < redec.length; i++){
         Densa camada = redec[i];

         this.m[i] = new Mat(camada.pesos.lin, camada.pesos.col);
         this.v[i] = new Mat(camada.pesos.lin, camada.pesos.col);
         
         if(camada.temBias()){
            this.mb[i] = new Mat(camada.bias.lin, camada.bias.col);
            this.vb[i] = new Mat(camada.bias.lin, camada.bias.col);
         }
      }
   }

   @Override
   public void atualizar(Densa[] redec){
      interacoes++;
      double forcaB1 = (1 - Math.pow(beta1, interacoes));
      double forcaB2 = (1 - Math.pow(beta2, interacoes));
   
      for(int i = 0; i < redec.length; i++){
         Densa camada = redec[i];
         Mat pesos = camada.pesos;
         Mat grads = camada.gradPesos;

         for(int j = 0; j < pesos.lin; j++){
            for(int k = 0; k < pesos.col; k++){
               calcular(pesos, grads, m[i], v[i], j, k, forcaB1, forcaB2);
            }
         }
         
         if(camada.temBias()){
            Mat bias = camada.bias;
            Mat gradsB = camada.gradSaida;
            for(int j = 0; j < bias.lin; j++){
               for(int k = 0; k < bias.col; k++){
                  calcular(bias, gradsB, mb[i], vb[i], j, k, forcaB1, forcaB2);
               }
            }
         }     
      }
   }

   private void calcular(Mat var, Mat grad, Mat m, Mat v, int lin, int col, double fb1, double fb2){
      double g = grad.dado(lin, col);

      double m2 = (beta1 * m.dado(lin, col)) + ((1 - beta1) * g);
      double v2 = (beta2 * v.dado(lin, col)) + ((1 - beta2) * (g*g));
      m.editar(lin, col, m2);
      v.editar(lin, col, v2);

      //correções
      double mChapeu = (beta1 * m.dado(lin, col) + ((1 - beta1) * g)) / fb1;
      double vChapeu = (beta2 * v.dado(lin, col)) / fb2;
      double att = (taxaAprendizagem * mChapeu) / (Math.sqrt(vChapeu) + epsilon);

      var.add(lin, col, att);
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
