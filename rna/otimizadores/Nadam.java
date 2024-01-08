package rna.otimizadores;

import rna.estrutura.Camada;

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
 *    {@code v} - variável que será otimizada (kernel, bias).
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
   private double[] m;
   
   /**
    * Coeficientes de momentum.
    */
   private double[] mb;

   /**
    * Coeficientes de momentum de segunda orgem.
    */
   private double[] v;

   /**
    * Coeficientes de momentum de segunda orgem.
    */
   private double[] vb;

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
   public void construir(Camada[] camadas){
      int nKernel = 0;
      int nBias = 0;
      
      for(Camada camada : camadas){
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
      this.construido = true;//otimizador pode ser usado
   }

   @Override
   public void atualizar(Camada[] camadas){
      super.verificarConstrucao();
      int idKernel = 0, idBias = 0;
      double g, mChapeu, vChapeu;

      interacoes++;
      double forcaB1 = 1 - Math.pow(beta1, interacoes);
      double forcaB2 = 1 - Math.pow(beta2, interacoes);
   
      for(Camada camada : camadas){
         if(camada.treinavel == false) continue;

         double[] kernel = camada.obterKernel();
         double[] gradK = camada.obterGradKernel();

         for(int i = 0; i < kernel.length; i++){
            g = gradK[i];
            m[idKernel] = (beta1 * m[idKernel]) + ((1 - beta1) * g);
            v[idKernel] = (beta2 * v[idKernel]) + ((1 - beta2) * (g*g));

            mChapeu = (beta1 * m[idKernel]) + ((1 - beta1) * g) / forcaB1;
            vChapeu = (beta2 * v[idKernel]) / forcaB2;
            kernel[i] += (taxaAprendizagem * mChapeu) / (Math.sqrt(vChapeu) + epsilon);
            idKernel++;
         }
         camada.editarKernel(kernel);
         
         if(camada.temBias()){
            double[] bias = camada.obterBias();
            double[] gradB = camada.obterGradBias();

            for(int i = 0; i < bias.length; i++){
               g = gradB[i];
               mb[idBias] = (beta1 * mb[idBias]) + ((1 - beta1) * g);
               vb[idBias] = (beta2 * vb[idBias]) + ((1 - beta2) * (g*g));

               mChapeu = (beta1 * mb[idBias]) + ((1 - beta1) * g) / forcaB1;
               vChapeu = (beta2 * vb[idBias]) / forcaB2;
               bias[i] += (taxaAprendizagem * mChapeu) / (Math.sqrt(vChapeu) + epsilon);
               idBias++;
            }
            camada.editarBias(bias);
         }     
      }
   }

   @Override
   public String info(){
      super.verificarConstrucao();
      super.construirInfo();
      
      super.addInfo("TaxaAprendizagem: " + this.taxaAprendizagem);
      super.addInfo("Beta1: " + this.beta1);
      super.addInfo("Beta2: " + this.beta2);
      super.addInfo("Epsilon: " + this.epsilon);

      return super.info();
   }

}
