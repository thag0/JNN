package rna.otimizadores;

import rna.core.Matriz;
import rna.estrutura.CamadaDensa;

/**
 * Implementa uma versão do algoritmo AdaGrad (Adaptive Gradient Algorithm).
 * O algoritmo otimiza o processo de aprendizado adaptando a taxa de aprendizagem 
 * de cada parâmetro com base no histórico de atualizações 
 * anteriores
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
   private double[][][] ac;

   /**
    * Acumuladores dos bias.
    */
   private double[][][] acb;

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
      this.ac = new double[redec.length][][];
      this.acb = new double[redec.length][][];
      double valorInicial = 0.1;

      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         this.ac[i] = new double[camada.pesos.lin][camada.pesos.col];
         mat.preencher(this.ac[i], valorInicial);

         if(camada.temBias()){
            this.acb[i] = new double[camada.bias.lin][camada.bias.col];
            mat.preencher(this.acb[i], valorInicial);
         }
      }
   }

   /**
    * Aplica o algoritmo do AdaGrad para cada peso da rede neural.
    * <p>
    *    O Adagrad funciona usando a seguinte expressão:
    * </p>
    * <pre>
    *    p[i] -= (tA * g[i]) / (√ ac[i] + eps)
    * </pre>
    * Onde:
    * <p>
    *    {@code p} - peso que será atualizado.
    * </p>
    * <p>
    *    {@code tA} - valor de taxa de aprendizagem (learning rate).
    * </p>
    * <p>
    *    {@code g} - gradiente correspondente a conexão do peso que será
    *    atualizado.
    * </p>
    * <p>
    *    {@code ac} - acumulador de gradiente correspondente a conexão
    *    do peso que será atualizado.
    * </p>
    * <p>
    *    {@code eps} - um valor pequeno para evitar divizões por zero.
    * </p>
    */
   @Override
   public void atualizar(CamadaDensa[] redec){
      double g;
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         for(int j = 0; j < camada.pesos.lin; j++){
            for(int k = 0; k < camada.pesos.col; k++){
               g = camada.gradientes.dado(j, k);
               ac[i][j][k] += g * g;
               camada.pesos.add(j, k, calcular(g, ac[i][j][k]));
            }
         }
         
         if(camada.temBias()){
            for(int j = 0; j < camada.bias.lin; j++){
               for(int k = 0; k < camada.bias.col; k++){
                  g = camada.erros.dado(j, k);
                  acb[i][j][k] += g * g;
                  camada.bias.add(j, k, calcular(g, acb[i][j][k]));
               }
            }
         }
      }
   }

   private double calcular(double g, double ac){
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
