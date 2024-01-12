package rna.otimizadores;

import rna.camadas.Camada;

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
 *v[i][j] -= m[i][j] // apenas com momentum
 *v[i][j] -= (-g[i][j] * tA) - (M * m[i][j]) // com nesterov
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada (kernel, bias).
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
    * @param momentum taxa de momentum do otimizador.
    * @param nesterov usar acelerador de nesterov.
    */
   public SGD(double tA, double momentum, boolean nesterov){
      if(tA <= 0 | tA > 1){
         throw new IllegalArgumentException(
            "O valor da taxa de aprendizagem deve estar entre ]0, 1], " + 
            "recebido: " + tA
         );
      }
      
      if(momentum < 0 | momentum > 1){         
         throw new IllegalArgumentException(
            "O valor de momentum deve estar entre [0, 1], " + 
            "recebido: " + momentum
         );
      }

      this.taxaAprendizagem = tA;
      this.momentum = momentum;
      this.nesterov = nesterov;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
    * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
    * @param tA taxa de aprendizagem do otimizador.
    * @param momentum taxa de momentum do otimizador.
    */
   public SGD(double tA, double momentum){
      this(tA, momentum, false);
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
    * Descent (SGD) </strong> usando os valores de hiperparâmetros fornecidos.
    * @param tA taxa de aprendizagem do otimizador.
    */
   public SGD(double tA){
      this(tA, 0, false);
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient 
    * Descent (SGD) </strong>.
    * <p>
    *    Os hiperparâmetros do SGD serão inicializados com seus os valores padrão.
    * </p>
    */
   public SGD(){
      this(0.01, 0.9, false);
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
      this.mb = new double[nBias];
      this.construido = true;//otimizador pode ser usado
   }

   @Override
   public void atualizar(Camada[] camadas){
      super.verificarConstrucao();

      int i, idKernel = 0, idBias = 0;

      for(Camada camada : camadas){
         if(camada.treinavel == false) continue;

         double[] kernel = camada.obterKernel();
         double[] gradK = camada.obterGradKernel();
         for(i = 0; i < kernel.length; i++){
            m[idKernel] = (m[idKernel] * momentum) - (gradK[i] * taxaAprendizagem);
            kernel[i] -= nesterov ? (m[idKernel++] * momentum) - (gradK[i] * taxaAprendizagem) : m[idKernel++];
         }
         camada.editarKernel(kernel);

         if(camada.temBias()){
            double[] bias = camada.obterBias();
            double[] gradB = camada.obterGradBias();
            for(i = 0; i < bias.length; i++){
               mb[idBias] = (mb[idBias] * momentum) - (gradB[i] * taxaAprendizagem);
               bias[i] -= nesterov ? (mb[idBias++] * momentum) - (gradB[i] * taxaAprendizagem) : mb[idBias++];
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
      super.addInfo("Momentum: " + this.momentum);
      super.addInfo("Nesterov: " + this.nesterov);

      return super.info();
   }

}
