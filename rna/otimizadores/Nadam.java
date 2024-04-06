package rna.otimizadores;

import rna.camadas.Camada;

/**
 * <h2>
 *    Nesterov-accelerated Adaptive Moment Estimation
 * </h2>
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
 *    v -= (tA * mc) / ((√ vc) + eps)
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
 *    {@code vc} - valor de velocidade (momentum de segunda ordem) corrigido
 * </p>
 * Os valores de momentum e velocidade corrigidos se dão por:
 * <pre>
 *mc = ((beta1 * m) + ((1 - beta1) * g)) / (1 - beta1ⁱ)
 *vc = (beta2 * v) / (1 - beta2ⁱ)
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

   /**
    * Valor padrão para a taxa de aprendizagem do otimizador.
    */
   private static final double PADRAO_TA = 0.001;

   /**
    * Valor padrão para o decaimento do momento de primeira ordem.
    */
   private static final double PADRAO_BETA1 = 0.9;

   /**
    * Valor padrão para o decaimento do momento de segunda ordem.
    */
   private static final double PADRAO_BETA2 = 0.999;
   
   /**
    * Valor padrão para epsilon.
    */
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
      if(tA <= 0){
         throw new IllegalArgumentException(
            "\nTaxa de aprendizagem (" + tA + "), inválida."
         );
      }
      if(beta1 <= 0){
         throw new IllegalArgumentException(
            "\nTaxa de decaimento de primeira ordem (" + beta1 + "), inválida."
         );
      }
      if(beta2 <= 0){
         throw new IllegalArgumentException(
            "\nTaxa de decaimento de segunda ordem (" + beta2 + "), inválida."
         );
      }
      if(epsilon <= 0){
         throw new IllegalArgumentException(
            "\nEpsilon (" + epsilon + "), inválido."
         );
      }
      
      this.taxaAprendizagem = tA;
      this.beta1 = beta1;
      this.beta2 = beta2;
      this.epsilon = epsilon;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Nadam </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
    * @param beta1 decaimento do momento de primeira ordem.
    * @param beta2 decaimento da segunda ordem.
    */
   public Nadam(double tA, double beta1, double beta2){
      this(tA, beta1, beta2, PADRAO_EPS);
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
         if(camada.treinavel() == false) continue;

         nKernel += camada.kernelParaArray().length;
         if(camada.temBias()){
            nBias += camada.biasParaArray().length;
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
      verificarConstrucao();
      
      interacoes++;
      double forcaB1 = 1 - Math.pow(beta1, interacoes);
      double forcaB2 = 1 - Math.pow(beta2, interacoes);
      
      int idKernel = 0, idBias = 0;
      for(Camada camada : camadas){
         if(camada.treinavel() == false) continue;

         double[] kernel = camada.kernelParaArray();
         double[] gradK = camada.gradKernelParaArray();
         idKernel = calcular(kernel, gradK, m, v, forcaB1, forcaB2, idKernel);
         camada.setKernel(kernel);
         
         if(camada.temBias()){
            double[] bias = camada.biasParaArray();
            double[] gradB = camada.gradBias();
            idBias = calcular(bias, gradB, mb, vb, forcaB1, forcaB2, idBias);
            camada.setBias(bias);
         }     
      }
   }

   /**
    * Atualiza as variáveis usando o gradiente pré calculado.
    * @param vars variáveis que serão atualizadas.
    * @param grads gradientes das variáveis.
    * @param m coeficientes de momentum de primeira ordem das variáveis.
    * @param v coeficientes de momentum de segunda ordem das variáveis.
    * @param forcaB1 força do decaimento do momentum de primeira ordem.
    * @param forcaB2 força do decaimento do momentum de segunda ordem.
    * @param id índice inicial das variáveis dentro do array de momentums.
    * @return índice final após as atualizações.
    */
   private int calcular(double[] vars, double[] grads, double[] m, double[] v, double forcaB1, double forcaB2, int id){
      double g, mChapeu, vChapeu;
      
      for(int i = 0; i < vars.length; i++){
         g = grads[i];
         m[id] = (beta1 * m[id]) + ((1 - beta1) * g);
         v[id] = (beta2 * v[id]) + ((1 - beta2) * (g*g));
         
         mChapeu = (beta1 * m[id]) + ((1 - beta1) * g) / forcaB1;
         vChapeu = (beta2 * v[id]) / forcaB2;
         vars[i] -= (mChapeu * taxaAprendizagem) / (Math.sqrt(vChapeu) + epsilon);
      
         id++;
      }

      return id;
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
