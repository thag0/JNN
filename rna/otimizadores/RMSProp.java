package rna.otimizadores;

import rna.estrutura.Camada;

/**
 * Implementa o treino da rede neural usando o algoritmo RMSProp (Root Mean Square Propagation).
 * <p>
 *    Ele é uma adaptação do Gradiente Descendente Estocástico (SGD) que ajuda a lidar com a
 *    oscilação do gradiente, permitindo que a taxa de aprendizado seja adaptada para cada 
 *    parâmetro individualmente.
 * </p>
 * <p>
 * 	Os hiperparâmetros do RMSProp podem ser ajustados para controlar 
 *    o comportamento do otimizador durante o treinamento.
 * </p>
 * <p>
 *    O RMSProp funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v[i][j] -= (-g[i][j] * tA) / ((√ ac[i][j]) + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada (kernel, bias).
 * </p>
 * <p>
 *    {@code g} - gradiente correspondente a variável
 *    que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - taxa de aprendizagem do otimizador.
 * </p>
 * <p>
 *    {@code ac} - acumulador de gradiente correspondente a variável
 *    que será otimizada.
 * </p>
 */
public class RMSProp extends Otimizador{
   private static final double PADRAO_TA  = 0.001;
   private static final double PADRAO_RHO = 0.995;
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
    * Fator de decaimento.
    */
   private double rho;

   /**
    * Acumuladores para os kernels
    */
   private double[] ac;

   /**
    * Acumuladores para os bias.
    */
   private double[] acb;

   /**
    * Inicializa uma nova instância de otimizador <strong> RMSProp </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
    * @param rho fator de decaimento do RMSProp.
    * @param epsilon usado para evitar a divisão por zero.
    */
   public RMSProp(double tA, double rho, double epsilon){
      this.taxaAprendizagem = tA;
      this.rho = rho;
      this.epsilon = epsilon;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> RMSProp </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
    */
   public RMSProp(double tA){
      this(tA, PADRAO_RHO, PADRAO_EPS);
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> RMSProp </strong>.
    * <p>
    *    Os hiperparâmetros do RMSProp serão inicializados com os valores padrão, que são:
    * </p>
    * <p>
    *    {@code taxaAprendizagem = 0.001}
    * </p>
    * <p>
    *    {@code rho = 0.99}
    * </p>
    * <p>
    *    {@code epsilon = 1e-7}
    * </p>
    */
   public RMSProp(){
      this(PADRAO_TA, PADRAO_RHO, PADRAO_EPS);
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

      this.ac  = new double[nKernel];
      this.acb = new double[nBias];
   }

   @Override
   public void atualizar(Camada[] redec){
      int idKernel = 0, idBias = 0;
      double g;

      for(Camada camada : redec){
         if(camada.treinavel == false) continue;

         double[] kernel = camada.obterKernel();
         double[] gradK = camada.obterGradKernel();

         for(int i = 0; i < kernel.length; i++){
            g = gradK[i];
            ac[idKernel] = (rho * ac[idKernel]) + ((1 - rho) * (g*g));
            kernel[i] += (g * taxaAprendizagem) / (Math.sqrt(ac[idKernel]) + epsilon);
            idKernel++;
         }
         camada.editarKernel(kernel);

         if(camada.temBias()){
            double[] bias = camada.obterBias();
            double[] gradB = camada.obterGradBias();
            
            for(int i = 0; i < bias.length; i++){
               g = gradB[i];
               acb[idBias] = (rho * acb[idBias]) + ((1 - rho) * (g*g));
               bias[i] += (g * taxaAprendizagem) / (Math.sqrt(acb[idBias]) + epsilon);
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
      buffer += espacamento + "Rho: " + this.rho + "\n";
      buffer += espacamento + "Epsilon: " + this.epsilon + "\n";

      return buffer;
   }

}
