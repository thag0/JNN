package rna.otimizadores;

import rna.core.Mat;
import rna.estrutura.CamadaDensa;

/**
 * Implementação do algoritmo de otimização Adam.
 * <p>
 *    O algoritmo ajusta os pesos da rede neural usando o gradiente descendente 
 *    com momento e a estimativa adaptativa de momentos de primeira e segunda ordem.
 * </p>
 * <p>
 * 	Os hiperparâmetros do Adam podem ser ajustados para controlar o 
 * 	comportamento do otimizador durante o treinamento.
 * </p>
 * <p>
 *    O Adam funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    var[i][j] -= (alfa * m[i][j]) / ((√ v[i][j]) + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code var} - variável que será otimizada (peso ou bias).
 * </p>
 * <p>
 *    {@code alfa} - correção aplicada a taxa de aprendizagem.
 * </p>
 * <p>
 *    {@code m} - coeficiente de momentum correspondente a variável que
 *    será otimizada;
 * </p>
 * <p>
 *    {@code v} - coeficiente de momentum de segunda orgem correspondente 
 *    a variável que será otimizada;
 * </p>
 * <p>
 *    {@code eps} - pequeno valor usado para evitar divisão por zero.
 * </p>
 * O valor de {@code alfa} é dado por:
 * <pre>
 * alfa = taxaAprendizagem * √(1- beta1ⁱ) / (1 - beta2ⁱ)
 * </pre>
 * Onde:
 * <p>
 *    {@code i} - contador de interações do Adam.
 * </p>
 * As atualizações de momentum de primeira e segunda ordem se dão por:
 *<pre>
 *m[i][j] += (1 - beta1) * (g  - m[i][j])
 *v[i][j] += (1 - beta2) * (g² - v[i][j])
 *</pre>
 * Onde:
 * <p>
 *    {@code beta1 e beta2} - valores de decaimento dos momentums de primeira
 *    e segunda ordem.
 * </p>
 * <p>
 *    {@code g} - gradiente correspondente a variável que será otimizada.
 * </p>
 */
public class Adam extends Otimizador{

   private static final double padraoTA = 0.001;
   private static final double padraoBeta1 = 0.9;
   private static final double padraoBeta2 = 0.999;
   private static final double padraoEps = 1e-7; 

   /**
    * Valor de taxa de aprendizagem do otimizador.
    */
   private double taxaAprendizagem;

   /**
    * Decaimento do momentum.
    */
   private double beta1;
    
   /**
    * Decaimento do momentum de segunda ordem.
    */
   private double beta2;
    
   /**
    * Usado para evitar divisão por zero.
    */
   private double epsilon;

   /**
    * Coeficientes de momentum.
    */
   private Mat[] m;

   /**
    * Coeficientes de momentum para os bias.
    */
   private Mat[] mb;

   /**
    * Coeficientes de momentum de segunda ordem.
    */
   private Mat[] v;

   /**
    * Coeficientes de momentum de segunda ordem para os bias.
    */
   private Mat[] vb;
   
   /**
    * Contador de iterações.
    */
   long interacoes = 0;
 
   /**
    * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA taxa de aprendizagem do otimizador.
    * @param beta1 decaimento do momento de primeira ordem.
    * @param beta2 decaimento do momento de segunda ordem.
    * @param epsilon usado para evitar a divisão por zero.
    */
   public Adam(double tA, double beta1, double beta2, double epsilon){
      this.taxaAprendizagem = tA;
      this.beta1 = beta1;
      this.beta2 = beta2;
      this.epsilon = epsilon;
   }
 
   /**
    * Inicializa uma nova instância de otimizador <strong> Adam </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA taxa de aprendizagem do otimizador.
    */
   public Adam(double tA){
      this(tA, padraoBeta1, padraoBeta2, padraoEps);
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Adam </strong>.
    * <p>
    *    Os hiperparâmetros do Adam serão inicializados com os valores 
    *    padrão, que são:
    * </p>
    * <p>
    *    {@code taxaAprendizagem = 0.001}
    * </p>
    * <p>
    *    {@code beta1 = 0.9}
    * </p>
    * <p>
    *    {@code beta2 = 0.999}
    * </p>
    * <p>
    *    {@code epsilon = 1e-7}
    * </p>
    */
   public Adam(){
      this(padraoTA, padraoBeta1, padraoBeta2, padraoEps);
   }

   @Override
   public void inicializar(CamadaDensa[] redec){
      this.m  = new Mat[redec.length];
      this.v  = new Mat[redec.length];
      this.mb = new Mat[redec.length];
      this.vb = new Mat[redec.length];
   
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         this.m[i] = new Mat(camada.pesos.lin, camada.pesos.col);
         this.v[i] = new Mat(camada.pesos.lin, camada.pesos.col);
         
         if(camada.temBias()){
            this.mb[i] = new Mat(camada.bias.lin, camada.bias.col);
            this.vb[i] = new Mat(camada.bias.lin, camada.bias.col);
         }
      }
   }

   @Override
   public void atualizar(CamadaDensa[] redec){
      interacoes++;
      double forcaB1 = Math.pow(beta1, interacoes);
      double forcaB2 = Math.pow(beta2, interacoes);
      double alfa = taxaAprendizagem * Math.sqrt(1 - forcaB2) / (1 - forcaB1);
   
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];
         Mat pesos = camada.pesos;
         Mat grads = camada.gradientes;

         for(int j = 0; j < camada.pesos.lin; j++){
            for(int k = 0; k < camada.pesos.col; k++){
               calcular(pesos, grads, m[i], v[i], j, k, alfa, forcaB1, forcaB2);
            }
         }
         
         if(camada.temBias()){
            Mat bias = camada.bias;
            Mat gradsB = camada.erros;
            for(int j = 0; j < bias.lin; j++){
               for(int k = 0; k < bias.col; k++){
                  calcular(bias, gradsB, mb[i], vb[i], j, k, alfa, forcaB1, forcaB2);
               }
            }
         }     
      }
   }

   private void calcular(Mat var, Mat grad, Mat m, Mat v, int lin, int col, double alfa, double fb1, double fb2){
      double g = grad.dado(lin, col);
      double d1 = (1 - beta1) * (g - m.dado(lin, col));
      double d2 = (1 - beta2) * ((g*g) - v.dado(lin, col)); 
      m.add(lin, col, d1); 
      v.add(lin, col, d2);

      double att = (alfa * m.dado(lin, col)) / (Math.sqrt(v.dado(lin, col)) + this.epsilon);
      var.add(lin, col, att);
   }

   @Override
   public String info(){
      String buffer = "";

      String espacamento = "    ";
      buffer += espacamento + "TaxaAprendizagem: " + this.taxaAprendizagem + "\n";
      buffer += espacamento + "Beta1: " + this.beta1 + "\n";
      buffer += espacamento + "Beta2: " + this.beta2 + "\n";
      buffer += espacamento + "Epsilon: " + this.epsilon + "\n";

      return buffer;
   }

}
