package rna.otimizadores;

import rna.estrutura.CamadaDensa;


/**
 * Implementação do algoritmo de otimização Nadam.
 * <p>
 *    O algoritmo ajusta os pesos da rede neural usando o gradiente descendente 
 *    com momento e a estimativa adaptativa de momentos de primeira e segunda ordem.
 * </p>
 * O adicional do Nadam é usar o acelerador de nesterov durante a correção dos
 * pesos da rede.
 */
public class Nadam extends Otimizador{

   /**
    * Valor de taxa de aprendizagem do otimizador.
    */
   private double taxaAprendizagem;

   /**
    * Usado para evitar divisão por zero.
    */
   private double epsilon;

   /**
    * decaimento do momentum.
    */
   private double beta1;

   /**
    * decaimento do momentum de segunda ordem.
    */
   private double beta2;

   /**
    * Coeficientes de momentum.
    */
   private double[][][] m;
   
   /**
    * Coeficientes de momentum.
    */
   private double[][][] mb;

   /**
    * Coeficientes de momentum de segunda orgem.
    */
   private double[][][] v;

   /**
    * Coeficientes de momentum de segunda orgem.
    */
   private double[][][] vb;

   /**
    * Contador de iterações.
    */
   long interacoes = 0;

   /**
    * Inicializa uma nova instância de otimizador <strong> Nadam </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
    * @param beta1 decaimento do momento de primeira ordem.
    * @param beta2 decaimento da segunda ordem.
    * @param epsilon usado para evitar a divisão por zero.
    */
   public Nadam(double tA, double beta1, double beta2, double epsilon){
      this.taxaAprendizagem = tA;
      this.beta1 = beta1;
      this.beta2 = beta2;
      this.epsilon = epsilon;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Nadam </strong>.
    * <p>
    *    Os hiperparâmetros do Nadam serão inicializados com os valores padrão, que são:
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
   public Nadam(){
      this(0.001, 0.9, 0.999, 1e-7);
   }

   @Override
   public void inicializar(CamadaDensa[] redec){
      this.m = new double[redec.length][][];
      this.v = new double[redec.length][][];
      this.mb = new double[redec.length][][];
      this.vb = new double[redec.length][][];
   
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         this.m[i] = new double[camada.pesos.lin][camada.pesos.col];
         this.v[i] = new double[camada.pesos.lin][camada.pesos.col];
         
         if(camada.temBias()){
            this.mb[i] = new double[camada.bias.lin][camada.bias.col];
            this.vb[i] = new double[camada.bias.lin][camada.bias.col];
         }
      }
   }

   /**
    * Aplica o algoritmo do Nadam para cada peso da rede neural.
    * <p>
    *    O Nadam funciona usando a seguinte expressão:
    * </p>
    * <pre>
    *    p[i] -= (tA * mc) / ((√ vc) + eps)
    * </pre>
    * Onde:
    * <p>
    *    {@code p} - peso que será atualizado.
    * </p>
    * <p>
    *    {@code tA} - valor de taxa de aprendizagem do otimizador.
    * </p>
    * <p>
    *    {@code mc} - valor de momentum corrigido
    * </p>
    * <p>
    *    {@code m2c} - valor de velocidade (momentum de segunda ordem) corrigido
    * </p>
    * Os valores de momentum e velocidade corrigidos se dão por:
    * <pre>
    *    mc = ((beta1 * m) + ((1 - beta1) * g[i])) / (1 - beta1ⁱ)
    * </pre>
    * <pre>
    *    vc = (beta2 * v) / (1 - beta2ⁱ)
    * </pre>
    * Onde:
    * <p>
    *    {@code m} - valor de momentum correspondete a conexão do peso que está
    *     sendo atualizado.
    * </p>
    * <p>
    *    {@code v} - valor de velocidade correspondete a conexão do peso que está 
    *    sendo atualizado.
    * </p>
    * <p>
    *    {@code g} - gradiente correspondente a conexão do peso que será
    *    atualizado.
    * </p>
    * <p>
    *    {@code i} - contador de interações (épocas passadas em que o otimizador foi usado) 
    * </p>
    */
   @Override
   public void atualizar(CamadaDensa[] redec){
      interacoes++;
      double g;
      double forcaB1 = (1 - Math.pow(beta1, interacoes));
      double forcaB2 = (1 - Math.pow(beta2, interacoes));
   
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         for(int j = 0; j < camada.pesos.lin; j++){
            for(int k = 0; k < camada.pesos.col; k++){
               g = camada.gradientes.dado(j, k);

               m[i][j][k] = (beta1 * m[i][j][k]) + ((1 - beta1) * g);
               v[i][j][k] = (beta2 * v[i][j][k]) + ((1 - beta2) * (g*g));

               camada.pesos.add(j, k, calcular(g, m[i][j][k], v[i][j][k], forcaB1, forcaB2));
            }
         }
         
         if(camada.temBias()){
            for(int j = 0; j < camada.bias.lin; j++){
               for(int k = 0; k < camada.bias.col; k++){
                  g = camada.erros.dado(j, k);

                  mb[i][j][k] = (beta1 * mb[i][j][k]) + ((1 - beta1) * g);
                  vb[i][j][k] = (beta2 * vb[i][j][k]) + ((1 - beta2) * (g*g));

                  camada.bias.add(j, k, calcular(g, mb[i][j][k], vb[i][j][k], forcaB1, forcaB2));
               }
            }
         }     
      }
   }

   private double calcular(double g, double m, double v, double forcaB1, double forcaB2){
      //correções
      double mChapeu = (beta1 * m + ((1 - beta1) * g)) / forcaB1;
      double vChapeu = (beta2 * v) / forcaB2;
      return (taxaAprendizagem * mChapeu) / (Math.sqrt(vChapeu) + epsilon);
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
