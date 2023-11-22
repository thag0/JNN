package rna.otimizadores;

import rna.estrutura.CamadaDensa;

/**
 * Implementa o treino da rede neural usando o algoritmo RMSProp (Root Mean Square Propagation).
 *
 * Ele é uma adaptação do Gradiente Descendente Estocástico (SGD) que ajuda a lidar com a
 * oscilação do gradiente, permitindo que a taxa de aprendizado seja adaptada para cada parâmetro 
 * individualmente.
 * <p>
 * 	Os hiperparâmetros do RMSProp podem ser ajustados para controlar 
 *    o comportamento do otimizador durante o treinamento.
 * </p
 */
public class RMSProp extends Otimizador{

   /**
    * Valor de taxa de aprendizagem do otimizador.
    */
   private double taxaAprendizagem;

   /**
    * Usado para evitar divisão por zero.
    */
   private double epsilon;
  
   /**
    * fator de decaimento do RMSprop.
    */
   private double rho;

   /**
    * Acumuladores para os pesos
    */
   private double[][][] ac;

   /**
    * Acumuladores para os bias.
    */
   private double[][][] acb;

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
      this(0.001, 0.99, 1e-7);
   }

   @Override
   public void inicializar(CamadaDensa[] redec){
      this.ac = new double[redec.length][][];
      this.acb = new double[redec.length][][];

      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         this.ac[i] = new double[camada.pesos.length][camada.pesos[0].length];
         if(camada.temBias()){
            acb[i] = new double[camada.bias.length][camada.bias[0].length];
         }
      }
   }

   /**
    * Aplica o algoritmo do RMSProp para cada peso da rede neural.
    * <p>
    *    O Nadam funciona usando a seguinte expressão:
    * </p>
    * <pre>
    *    p[i] -= tA / ((√ ac[i]) + eps) * g[i]
    * </pre>
    * Onde:
    * <p>
    *    {@code p} - peso que será atualizado.
    * </p>
    * <p>
    *    {@code tA} - valor de taxa de aprendizagem (learning rate).
    * </p>
    * <p>
    *    {@code ac} - acumulador de gradiente correspondente a conexão do 
    *    peso que será atualizado.
    * </p>
    * <p>
    *    {@code g} - gradiente correspondente a conexão do peso que será
    *    atualizado.
    * </p>
    */
   @Override
   public void atualizar(CamadaDensa[] redec){
      //TODO 
      //resolver problema de convergência, rede não aprendendo nada
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         for(int j = 0; j < camada.pesos.length; j++){
            for(int k = 0; k < camada.pesos[j].length; k++){
               double grad = camada.gradientes[j][k];
               ac[i][j][k] += (rho * ac[i][j][k]) + (1 - rho) * (grad * grad);
               // camada.pesos[j][k] -= calcular(grad, ac[i][j][k]);
               camada.pesos[j][k] += (taxaAprendizagem * grad) / (Math.sqrt(ac[i][j][k] + epsilon));
            }
         }
         
         if(camada.temBias()){
            for(int j = 0; j < camada.bias.length; j++){
               for(int k = 0; k < camada.bias[j].length; k++){
                  double grad = camada.erros[j][k];
                  acb[i][j][k] += (rho * acb[i][j][k]) + (1 - rho) * (grad * grad);
                  // camada.bias[j][k] -= calcular(grad, acb[i][j][k]);
                  camada.bias[j][k] += (taxaAprendizagem * grad) / (Math.sqrt(acb[i][j][k] + epsilon));
               }
            }
         }
      }
   }

   private double calcular(double grad, double ac){
      return (taxaAprendizagem * grad) / (Math.sqrt(ac + epsilon));
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
