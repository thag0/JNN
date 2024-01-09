package rna.otimizadores;

import rna.camadas.Camada;

/**
 * Classe que implementa o otimizador Gradiente Descentente Estocástico com momentum.
 * <p>
 *    Também possui o adicional do acelerador de nesterov, mas deve ser configurado.
 * </p>
 * <p>
 *    O SGD funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v[i][j] -= (-g[i][j] * tA) - (M * m[i][j])
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
    * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient Descent (SGD) </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA taxa de aprendizagem do otimizador.
    * @param momentum taxa de momentum do otimizador.
    * @param nesterov usar acelerador de nesterov.
    */
   public SGD(double tA, double momentum, boolean nesterov){
      this.taxaAprendizagem = tA;
      this.momentum = momentum;
      this.nesterov = nesterov;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient Descent (SGD) </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA taxa de aprendizagem do otimizador.
    * @param momentum taxa de momentum do otimizador.
    */
   public SGD(double tA, double momentum){
      this.taxaAprendizagem = tA;
      this.momentum = momentum;
      this.nesterov = false;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient Descent (SGD) </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA taxa de aprendizagem do otimizador.
    */
   public SGD(double tA){
      this.taxaAprendizagem = tA;
      this.momentum = 0;
      this.nesterov = false;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient Descent (SGD) </strong>.
    * <p>
    *    Os hiperparâmetros do SGD serão inicializados com seus os valores padrão.
    * </p>
    */
   public SGD(){
      this(0.001, 0.99, false);
   }

   @Override
   public void construir(Camada[] redec){
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
      this.mb = new double[nBias];
      this.construido = true;//otimizador pode ser usado
   }

   @Override
   public void atualizar(Camada[] redec){
      super.verificarConstrucao();

      int i, idKernel = 0, idBias = 0;

      for(Camada camada : redec){
         if(camada.treinavel == false) continue;

         double[] kernel = camada.obterKernel();
         double[] gradK = camada.obterGradKernel();

         for(i = 0; i < kernel.length; i++){
            m[idKernel] = (m[idKernel] * momentum) - (gradK[i] * taxaAprendizagem);

            if(nesterov){
               kernel[i] -= (m[idKernel] * momentum) - (gradK[i] * taxaAprendizagem);
            }else{
               kernel[i] -= m[idKernel];
            }

            idKernel++;
         }
         camada.editarKernel(kernel);

         if(camada.temBias()){
            double[] bias = camada.obterBias();
            double[] gradB = camada.obterGradBias();
            for(i = 0; i < bias.length; i++){
               mb[idBias] = (mb[idBias] * momentum) - (gradB[i] * taxaAprendizagem);

               if(nesterov){
                  bias[i] -= (mb[idBias] * momentum) - (gradB[i] * taxaAprendizagem);
               }else{
                  bias[i] -= mb[idBias];
               }

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
      super.addInfo("Momentum: " + this.momentum);
      super.addInfo("Nesterov: " + this.nesterov);

      return super.info();
   }

}
