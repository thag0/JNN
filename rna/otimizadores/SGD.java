package rna.otimizadores;

import rna.core.Mat;
import rna.estrutura.CamadaDensa;

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
 *    {@code v} - variável que será otimizada (peso ou bias).
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
    * Coeficientes de momentum para os pesos.
    */
   public Mat[] m;
   
   /**
    * Coeficientes de momentum para os bias.
    */
   public Mat[] mb;

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
   public void inicializar(CamadaDensa[] redec){
      this.m  = new Mat[redec.length];
      this.mb = new Mat[redec.length];

      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         this.m[i] = new Mat(camada.pesos.lin, camada.pesos.col);
         if(camada.temBias()){
            this.mb[i] = new Mat(camada.bias.lin, camada.bias.col);
         }
      }
   }

   @Override
   public void atualizar(CamadaDensa[] redec){
      int i, j, k;
      for(i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];
         Mat pesos = camada.pesos;
         Mat gradP = camada.gradientePesos;

         for(j = 0; j < pesos.lin; j++){
            for(k = 0; k < pesos.col; k++){
               calcular(pesos, gradP, m[i], j, k);
            }
         }

         if(camada.temBias()){
            Mat bias = camada.bias;
            Mat gradB = camada.gradienteBias;
            for(j = 0; j < bias.lin; j++){
               for(k = 0; k < bias.col; k++){
                  calcular(bias, gradB, mb[i], j, k);
               }
            }
         }
      }
   }

   private void calcular(Mat var, Mat grad, Mat m, int lin, int col){
      double att = (-grad.dado(lin, col) * this.taxaAprendizagem) + (m.dado(lin, col) * this.momentum);
      m.editar(lin, col, att);

      if(this.nesterov){
         double nest = (grad.dado(lin, col) * this.taxaAprendizagem) + (this.momentum * m.dado(lin, col));
         var.sub(lin, col, nest);

      }else{
         var.sub(lin, col, m.dado(lin, col));
      }
   }

   @Override
   public String info(){
      String buffer = "";

      String espacamento = "    ";
      buffer += espacamento + "TaxaAprendizagem: " + this.taxaAprendizagem + "\n";
      buffer += espacamento + "Momentum: " + this.momentum + "\n";
      buffer += espacamento + "Nesterov: " + this.nesterov + "\n";

      return buffer;
   }

}
