package rna.otimizadores;

import rna.core.Mat;
import rna.estrutura.Densa;

/**
 * Implementação do otimizador Adadelta.
 * <p>
 * 	Os hiperparâmetros do Adadelta podem ser ajustados para controlar o 
 * 	comportamento do otimizador durante o treinamento.
 * </p>
 * <p>
 *    O Adadelta funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v[i][j] -= delta
 * </pre>
 * Onde delta é dado por:
 * <pre>
 * delta = √(acAt[i][j] + eps) / √(ac[i][j] + eps) * g
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada (peso ou bias).
 * </p>
 * <p>
 *    {@code acAt} - acumulador atualizado correspondente a variável que
 *    será otimizada.
 * </p>
 * <p>
 *    {@code ac} - acumulador correspondente a variável que
 *    será otimizada
 * </p>
 * <p>
 *    {@code g} - gradientes correspondente a variável que será otimizada.
 * </p>
 * Os valores do acumulador (ac) e acumulador atualizado (acAt) se dão por:
 * <pre>
 *ac[i][j]   = (rho * ac[i][j])   + ((1 - rho) * g²)
 *acAt[i][j] = (rho * acAt[i][j]) + ((1 - rho) * delta²)
 * </pre>
 * Onde:
 * <p>
 *    {@code rho} - constante de decaimento do otimizador.
 * </p>
 */
public class Adadelta extends Otimizador{
   private static final double PADRAO_RHO = 0.99;
   private static final double PADRAO_EPS = 1e-7;

   /**
    * Constante de decaimento do otimizador.
    */
   private double rho;

   /**
    * Valor usado para evitar divisão por zero.
    */
   private double epsilon;

   /**
    * Acumuladores para os pesos.
    */
   private Mat[] ac;

   /**
    * Acumuladores para os bias.
    */
   private Mat[] acb;

   /**
    * Acumulador atualziado para os pesos.
    */
   private Mat[] acAt;

   /**
    * Acumulador atualizado para os bias.
    */
   private Mat[] acAtb;

   /**
    * Inicializa uma nova instância de otimizador <strong> Adadelta </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param rho valor de decaimento do otimizador.
    * @param epsilon usado para evitar a divisão por zero.
    */
   public Adadelta(double rho, double epsilon){
      this.rho = rho;
      this.epsilon = epsilon;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Adadelta </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param rho valor de decaimento do otimizador.
    * @param epsilon usado para evitar a divisão por zero.
    */
   public Adadelta(double rho){
      this(rho, PADRAO_EPS);
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Adadelta </strong>.
    * <p>
    *    Os hiperparâmetros do Adadelta serão inicializados com os valores padrão.
    * </p>
    */
   public Adadelta(){
      this(PADRAO_RHO, PADRAO_EPS);
   }

   @Override
   public void inicializar(Densa[] redec){
      this.ac   = new Mat[redec.length];
      this.acAt = new Mat[redec.length];

      this.acb   = new Mat[redec.length];
      this.acAtb = new Mat[redec.length];
   
      for(int i = 0; i < redec.length; i++){
         Densa camada = redec[i];

         this.ac[i]   = new Mat(camada.pesos.lin, camada.pesos.col);
         this.acAt[i] = new Mat(camada.pesos.lin, camada.pesos.col);
         
         if(camada.temBias()){
            this.acb[i]   = new Mat(camada.bias.lin, camada.bias.col);
            this.acAtb[i] = new Mat(camada.bias.lin, camada.bias.col);
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
               calcular(pesos, grads, ac[i], acAt[i], j, k);
            }
         }

         if(camada.temBias()){
            Mat bias = camada.bias;
            Mat gradsB = camada.gradienteSaida;
            for(int j = 0; j < bias.lin; j++){
               for(int k = 0; k < bias.col; k++){
                  calcular(bias, gradsB, acb[i], acAtb[i], j, k);
               }
            }
         }
      }
   }

   private void calcular(Mat var, Mat grad, Mat ac, Mat acAt, int lin, int col){
      double g = grad.dado(lin, col);
      double ac2 = (rho * ac.dado(lin, col)) + ((1 - rho) * (g*g));
      ac.editar(lin, col, ac2);

      double delta = Math.sqrt(acAt.dado(lin, col) + epsilon) / Math.sqrt(ac.dado(lin, col) + epsilon) * g;
      double acAt2 = (rho * acAt.dado(lin, col)) + ((1 - rho) * (delta*delta));
      acAt.editar(lin, col, acAt2);
      
      var.add(lin, col, delta);
   }

   @Override
   public String info(){
      String buffer = "";

      String espacamento = "    ";
      buffer += espacamento + "Rho: " + this.rho + "\n";
      buffer += espacamento + "Epsilon: " + this.epsilon + "\n";

      return buffer;
   }
}
