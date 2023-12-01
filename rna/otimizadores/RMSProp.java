package rna.otimizadores;

import rna.core.Mat;
import rna.estrutura.Densa;

/**
 * Implementa o treino da rede neural usando o algoritmo RMSProp (Root Mean Square Propagation).
 *
 * Ele é uma adaptação do Gradiente Descendente Estocástico (SGD) que ajuda a lidar com a
 * oscilação do gradiente, permitindo que a taxa de aprendizado seja adaptada para cada parâmetro 
 * individualmente.
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
 *    {@code v} - variável que será otimizada.
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
   private static final double PADRAO_RHO = 0.999;
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
    * Acumuladores para os pesos
    */
   private Mat[] ac;

   /**
    * Acumuladores para os bias.
    */
   private Mat[] acb;

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
   public void inicializar(Densa[] redec){
      this.ac  = new Mat[redec.length];
      this.acb = new Mat[redec.length];

      for(int i = 0; i < redec.length; i++){
         Densa camada = redec[i];

         this.ac[i] = new Mat(camada.pesos.lin, camada.pesos.col);
         if(camada.temBias()){
            this.acb[i] = new Mat(camada.bias.lin, camada.bias.col);
         }
      }
   }

   @Override
   public void atualizar(Densa[] redec){
      for(int i = 0; i < redec.length; i++){
         Densa camada = redec[i];
         Mat pesos = camada.pesos;
         Mat grads = camada.gradPesos;

         for(int j = 0; j < pesos.lin; j++){
            for(int k = 0; k < pesos.col; k++){
               calcular(pesos, grads, ac[i], j, k);
            }
         }

         if(camada.temBias()){
            Mat bias = camada.bias;
            Mat gradsB = camada.gradienteSaida;
            for(int j = 0; j < bias.lin; j++){
               for(int k = 0; k < bias.col; k++){
                  calcular(bias, gradsB, acb[i], j, k);
               }
            }
         }
      }
   }

   private void calcular(Mat var, Mat grad, Mat ac, int lin, int col){
      double g = grad.dado(lin, col);

      double ac2 = (rho * ac.dado(lin, col)) + (1 - rho) * (g*g);
      ac.editar(lin, col, ac2);
      
      double att = (g * this.taxaAprendizagem) / (Math.sqrt(ac2 + this.epsilon));
      var.sub(lin, col, -att);
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
