package rna.otimizadores;

import rna.core.Mat;
import rna.core.Matriz;
import rna.estrutura.CamadaDensa;

/**
 * Implementa uma versão do algoritmo AdaGrad (Adaptive Gradient Algorithm).
 * O algoritmo otimiza o processo de aprendizado adaptando a taxa de aprendizagem 
 * de cada parâmetro com base no histórico de atualizações 
 * anteriores.
 * <p>
 *    O Adagrad funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v[i][j] -= (tA * g[i][j]) / (√ ac[i][j] + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada (peso ou bias).
 * </p>
 * <p>
 *    {@code tA} - taxa de aprendizagem do otimizador.
 * </p>
 * <p>
 *    {@code g} - gradientes correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code ac} - acumulador de gradiente correspondente a variável que
 *    será otimizada.d
 * </p>
 * <p>
 *    {@code eps} - um valor pequeno para evitar divizões por zero.
 * </p>
 */
public class AdaGrad extends Otimizador{

   Matriz mat = new Matriz();

   /**
    * Valor de taxa de aprendizagem do otimizador.
    */
   private double taxaAprendizagem;

   /**
    * Usado para evitar divisão por zero.
    */
   private double epsilon;

   /**
    * Acumuladores dos gradientes ao quadrado.
    */
   private Mat[] ac;

   /**
    * Acumuladores dos bias.
    */
   private Mat[] acb;

   /**
    * Inicializa uma nova instância de otimizador <strong> AdaGrad </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
    * @param epsilon usado para evitar a divisão por zero.
    */
   public AdaGrad(double tA, double epsilon){
      this.taxaAprendizagem = tA;
      this.epsilon = epsilon;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> AdaGrad </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
    */
   public AdaGrad(double tA){
      this(tA, 1e-7);
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> AdaGrad </strong>.
    * <p>
    *    Os hiperparâmetros do AdaGrad serão inicializados com os valores padrão, que são:
    * </p>
    * <p>
    *    {@code taxaAprendizagem = 0.01}
    * </p>
    * <p>
    *    {@code epsilon = 1e-7}
    * </p>
    */
   public AdaGrad(){
      this(0.01, 1e-7);
   }

   @Override
   public void inicializar(CamadaDensa[] redec){
      this.ac  = new Mat[redec.length];
      this.acb = new Mat[redec.length];
      double valorInicial = 0.1;

      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         this.ac[i] = new Mat(camada.pesos.lin, camada.pesos.col);
         mat.preencher(this.ac[i], valorInicial);

         if(camada.temBias()){
            this.acb[i] = new Mat(camada.bias.lin, camada.bias.col);
            mat.preencher(this.acb[i], valorInicial);
         }
      }
   }

   @Override
   public void atualizar(CamadaDensa[] redec){
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];
         Mat pesos = camada.pesos;
         Mat grads = camada.gradientePesos;

         for(int j = 0; j < pesos.lin; j++){
            for(int k = 0; k < pesos.col; k++){
               calcular(pesos, grads, ac[i], j, k);
            }
         }
         
         if(camada.temBias()){
            Mat bias = camada.bias;
            Mat gradsB = camada.gradientes;
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
      ac.add(lin, col, g*g);
      var.add(lin, col, aplicarGradiente(g, ac.dado(lin, col)));
   }

   private double aplicarGradiente(double g, double ac){
      return (taxaAprendizagem * g) / (Math.sqrt(ac + epsilon));
   }

   @Override
   public String info(){
      String buffer = "";

      String espacamento = "    ";
      buffer += espacamento + "TaxaAprendizagem: " + this.taxaAprendizagem + "\n";
      buffer += espacamento + "Epsilon: " + this.epsilon + "\n";

      return buffer;
   }
   
}
