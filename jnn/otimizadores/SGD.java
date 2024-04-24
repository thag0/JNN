package jnn.otimizadores;

import jnn.camadas.Camada;

/**
 * <h2>
 *    Stochastic Gradient Descent 
 * </h2>
 * <p>
 *    Implementação do otimizador do gradiente estocástico com momentum e
 *    acelerador de nesterov.
 * </p>
 * <p>
 *    O SGD funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *m = (m * M) - (g * tA)
 *v += m // apenas com momentum
 *v += (M * m) - (g * tA) // com nesterov
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada.
 * </p>
 * <p>
 *    {@code M} - valor de taxa de momentum (ou constante de momentum) 
 *    do otimizador.
 * </p>
 * <p>
 *    {@code m} - valor de momentum da correspondente a variável que será
 *    otimizada.
 * </p>
 * <p>
 *    {@code g} - gradientes correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - taxa de aprendizagem do otimizador.
 * </p>
 */
public class SGD extends Otimizador{

   /**
    * Taxa de aprendizagem padrão do otimizador.
    */
   private static final double PADRAO_TA = 0.01;

   /**
    * Taxa de momentum padrão do otimizador.
    */
   private static final double PADRAO_MOMENTUM = 0.9;

   /**
    * Uso do acelerador de nesterov padrão.
    */
   private static final boolean PADRAO_NESTEROV = false;

   /**
    * Valor de taxa de aprendizagem do otimizador.
    */
   private double taxaAprendizagem;

   /**
    * Valor de taxa de momentum do otimizador.
    */
   private double momentum;

   /**
    * Usar acelerador de Nesterov.
    */
   private boolean nesterov;

   /**
    * Coeficientes de momentum para os kernels.
    */
   public double[] m;
   
   /**
    * Coeficientes de momentum para os bias.
    */
   public double[] mb;

   /**
    * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
    * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
    * @param tA taxa de aprendizagem do otimizador.
    * @param m taxa de momentum do otimizador.
    * @param nesterov usar acelerador de nesterov.
    */
   public SGD(double tA, double m, boolean nesterov){
      if(tA <= 0){
         throw new IllegalArgumentException(
            "\nTaxa de aprendizagem (" + tA + "), inválida."
         );
      }
      
      if(m < 0){         
         throw new IllegalArgumentException(
            "\nTaxa de momentum (" + m + "), inválida."
         );
      }

      this.taxaAprendizagem = tA;
      this.momentum = m;
      this.nesterov = nesterov;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
    * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
    * @param tA taxa de aprendizagem do otimizador.
    * @param m taxa de momentum do otimizador.
    */
   public SGD(double tA, double m){
      this(tA, m, PADRAO_NESTEROV);
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
    * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
    * @param tA taxa de aprendizagem do otimizador.
    */
   public SGD(double tA){
      this(tA, PADRAO_MOMENTUM, PADRAO_NESTEROV);
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
    * Descent (SGD) </strong>.
    * <p>
    *    Os hiperparâmetros do SGD serão inicializados com seus os valores padrão.
    * </p>
    */
   public SGD(){
      this(PADRAO_TA, PADRAO_MOMENTUM, PADRAO_NESTEROV);
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
      this.mb = new double[nBias];
      this._construido = true;//otimizador pode ser usado
   }

   @Override
   public void atualizar(Camada[] camadas){
      verificarConstrucao();

      int idKernel = 0, idBias = 0;
      for(Camada camada : camadas){
         if(camada.treinavel() == false) continue;

         double[] kernel = camada.kernelParaArray();
         double[] gradK = camada.gradKernelParaArray();
         idKernel = calcular(kernel, gradK, m, idKernel);
         camada.setKernel(kernel);

         if(camada.temBias()){
            double[] bias = camada.biasParaArray();
            double[] gradB = camada.gradBias();
            idBias = calcular(bias, gradB, mb, idBias);
            camada.setBias(bias);
         }
      }
   }

   /**
    * Atualiza as variáveis usando o gradiente pré calculado.
    * @param vars variáveis que serão atualizadas.
    * @param grads gradientes das variáveis.
    * @param m coeficientes de momentum.
    * @param id índice inicial das variáveis dentro do array de momentums.
    * @return índice final após as atualizações.
    */
   private int calcular(double[] vars, double[] grads, double[] m, int id){
      for(int i = 0; i < vars.length; i++){
         m[id] = (m[id] * momentum) - (grads[i] * taxaAprendizagem);
         vars[i] += nesterov ? (m[id] * momentum) - (grads[i] * taxaAprendizagem) : m[id];
         id++;
      }

      return id;
   }

   @Override
   public String info(){
      super.verificarConstrucao();
      super.construirInfo();
      
      super.addInfo("TaxaAprendizagem: " + this.taxaAprendizagem);
      super.addInfo("Momentum: " + this.momentum);
      super.addInfo("Nesterov: " + this.nesterov);

      return super.info();
   }

}
