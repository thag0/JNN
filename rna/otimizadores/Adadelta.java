package rna.otimizadores;

import rna.estrutura.Camada;

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
 *    {@code v} - variável que será otimizada (kernel, bias).
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
   private double[] ac;

   /**
    * Acumuladores para os bias.
    */
   private double[] acb;

   /**
    * Acumulador atualziado para os pesos.
    */
   private double[] acAt;

   /**
    * Acumulador atualizado para os bias.
    */
   private double[] acAtb;

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
      this.acAt  = new double[nKernel];
      this.acb = new double[nBias];
      this.acAtb = new double[nBias];
   }

   @Override
   public void atualizar(Camada[] redec){
      int idKernel = 0, idBias = 0;
      double g, delta;

      for(Camada camada : redec){
         if(camada.treinavel == false) continue;

         double[] kernel = camada.obterKernel();
         double[] gradK = camada.obterGradKernel();

         for(int i = 0; i < kernel.length; i++){
            g = gradK[i];
            ac[idKernel] = (rho * ac[idKernel]) + ((1 - rho) * (g*g));
            delta = Math.sqrt(acAt[idKernel] + epsilon) / Math.sqrt(ac[idKernel] + epsilon) * g;
            acAt[idKernel] = (rho * acAt[idKernel]) + ((1 - rho) * (delta * delta));
            kernel[i] += delta;

            idKernel++;
         }
         camada.editarKernel(kernel);

         if(camada.temBias()){
            double[] bias = camada.obterBias();
            double[] gradB = camada.obterGradBias();

            for(int i = 0; i < bias.length; i++){
               g = gradB[i];
               acb[idBias] = (rho * acb[idBias]) + ((1 - rho) * (g*g));
               delta = Math.sqrt(acAtb[idBias] + epsilon) / Math.sqrt(acb[idBias] + epsilon) * g;
               acAtb[idBias] = (rho * acAtb[idBias]) + ((1 - rho) * (delta * delta));
               bias[i] += delta;

               idBias++;       
            }
            camada.editarBias(bias);
         }
      }
   }

   @Override
   public String info(){
      String espacamento = "    ";
      
      String buffer = "";
      buffer += espacamento + "Otimizador: " +  this.nome() + "\n";
      buffer += espacamento + "Rho: " + this.rho + "\n";
      buffer += espacamento + "Epsilon: " + this.epsilon + "\n";

      return buffer;
   }
}
