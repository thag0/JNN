package rna.otimizadores;

import rna.estrutura.CamadaDensa;

/**
 * Classe que implementa o otimizador Gradiente Descentente Estocástico com momentum.
 * <p>
 *    Também possui o adicional do acelerador de nesterov, mas deve ser configurado.
 * </p>
 * Esse é o otimizador que me deu os melhores resultados de convergência até agora.
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
   public double[][][] m;
   
   /**
    * Coeficientes de momentum para os bias.
    */
   public double[][][] mb;

   /**
    * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient Descent (SGD) </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
    * @param momentum valor de taxa de momentum.
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
    * @param tA valor de taxa de aprendizagem.
    * @param momentum valor de taxa de momentum.
    */
   public SGD(double tA, double momentum){
      this.taxaAprendizagem = tA;
      this.momentum = momentum;
      this.nesterov = false;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Stochastic Gradient Descent (SGD) </strong>.
    * <p>
    *    Os hiperparâmetros do SGD serão inicializados com seus os valores padrão, que são:
    * </p>
    * <p>
    *    {@code taxaAprendizagem = 0.001}
    * </p>
    * <p>
    *    {@code momentum = 0.99}
    * </p>
    * <p>
    *    {@code nesterov = false}
    * </p>
    */
   public SGD(){
      this(0.001, 0.99, false);
   }

   @Override
   public void inicializar(CamadaDensa[] redec){
      this.m = new double[redec.length][][];
      this.mb = new double[redec.length][][];

      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         this.m[i] = new double[camada.pesos.length][camada.pesos[0].length];
         if(camada.temBias()){
            this.mb[i] = new double[camada.bias.length][camada.bias[0].length];
         }
      }
   }

   /**
    * Aplica o algoritmo do SGD com momentum (e nesterov, se configurado) para cada peso 
    * da rede neural.
    * <p>
    *    O SGD funciona usando a seguinte expressão:
    * </p>
    * <pre>
    *    p[i] -= (M * m[i]) + (g[i] * tA)
    * </pre>
    * Onde:
    * <p>
    *    {@code p} - peso que será atualizado.
    * </p>
    * <p>
    *    {@code M} - valor de taxa de momentum (ou constante de momentum) 
    *    do otimizador.
    * </p>
    * <p>
    *    {@code m} - valor de momentum da conexão correspondente ao peso
    *    que será atualizado.
    * </p>
    * <p>
    *    {@code g} - gradiente correspondente a conexão do peso que será
    *    atualizado.
    * </p>
    * <p>
    *    {@code tA} - taxa de aprendizagem do otimizador.
    * </p>
    */
   @Override
   public void atualizar(CamadaDensa[] redec){
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         for(int j = 0; j < camada.pesos.length; j++){
            for(int k = 0; k < camada.pesos[j].length; k++){
               m[i][j][k] = calcular(m[i][j][k], camada.gradientes[j][k]);
               
               if(nesterov){
                  camada.pesos[j][k] -= (camada.gradientes[j][k] * taxaAprendizagem) + (momentum * m[i][j][k]);
               }else{
                  camada.pesos[j][k] -= m[i][j][k];
               }
            }
         }

         if(camada.temBias()){
            for(int j = 0; j < camada.bias.length; j++){
               for(int k = 0; k < camada.bias[j].length; k++){
                  mb[i][j][k] = calcular(mb[i][j][k], camada.erros[j][k]);
                  
                  if(nesterov){
                     camada.bias[j][k] -= (camada.erros[j][k] * taxaAprendizagem) + (momentum * mb[i][j][k]);
                  }else{
                     camada.bias[j][k] -= mb[j][j][k];
                  }
               }
            }
         }
      }
   }

   private double calcular(double m, double grad){
      return (-grad * this.taxaAprendizagem) + (m * this.momentum);
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
