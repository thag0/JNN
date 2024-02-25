package rna.otimizadores;

import rna.camadas.Camada;

/**
 * <h2>
 *    Root Mean Square Propagation
 * </h2>
 * <p>
 *    Ele é uma adaptação do Gradiente Descendente Estocástico (SGD) que ajuda a lidar com a
 *    oscilação do gradiente, permitindo que a taxa de aprendizado seja adaptada para cada 
 *    parâmetro individualmente.
 * </p>
 * <p>
 * 	Os hiperparâmetros do RMSProp podem ser ajustados para controlar 
 *    o comportamento do otimizador durante o treinamento.
 * </p>
 * <p>
 *    O RMSProp funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *ac = (rho * ac) + ((1- rho) * g²);
 *v -= (g * tA) / ((√ ac) + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada..
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
 * <p>
 *    {@code rho} - taxa de decaimento do otimizador.
 * </p>
 */
public class RMSProp extends Otimizador{

   /**
    * Valor padrão para a taxa de aprendizagem do otimizador.
    */
   private static final double PADRAO_TA  = 0.001;

   /**
    * Valor padrão para a taxa de decaimeto.
    */
   private static final double PADRAO_RHO = 0.995;

	/**
	 * Valor padrão para epsilon.
	 */
	private static final double PADRAO_EPS = 1e-8;

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
    * Acumuladores para os kernels
    */
   private double[] ac;

   /**
    * Acumuladores para os bias.
    */
   private double[] acb;

   /**
    * Inicializa uma nova instância de otimizador <strong> RMSProp </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
    * @param rho fator de decaimento do RMSProp.
    * @param epsilon usado para evitar a divisão por zero.
    */
   public RMSProp(double tA, double rho, double epsilon){
      if(tA <= 0){
         throw new IllegalArgumentException(
            "\nTaxa de aprendizagem (" + tA + "), inválida."
         );
      }
      if(rho <= 0){
         throw new IllegalArgumentException(
            "\nTaxa de decaimento (" + rho + "), inválida."
         );
      }
      if(epsilon <= 0){
         throw new IllegalArgumentException(
            "\nEpsilon (" + epsilon + "), inválido."
         );
      }

      this.taxaAprendizagem = tA;
      this.rho = rho;
      this.epsilon = epsilon;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> RMSProp </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
    * @param rho fator de decaimento do RMSProp.
    */
   public RMSProp(double tA, double rho){
      this(tA, rho, PADRAO_EPS);
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
    *    Os hiperparâmetros do RMSProp serão inicializados com os valores padrão.
    * </p>
    */
   public RMSProp(){
      this(PADRAO_TA, PADRAO_RHO, PADRAO_EPS);
   }

   @Override
   public void construir(Camada[] camadas){
      int nKernel = 0;
      int nBias = 0;
      
      for(Camada camada : camadas){
         if(camada.treinavel == false) continue;

         nKernel += camada.kernel().length;
         if(camada.temBias()){
            nBias += camada.bias().length;
         }         
      }

      this.ac  = new double[nKernel];
      this.acb = new double[nBias];
      this.construido = true;//otimizador pode ser usado
   }

   @Override
   public void atualizar(Camada[] camadas){
      verificarConstrucao();
      
      int idKernel = 0, idBias = 0;
      for(Camada camada : camadas){
         if(camada.treinavel == false) continue;

         double[] kernel = camada.kernel();
         double[] gradK = camada.gradKernel();
         idKernel = calcular(kernel, gradK, ac, idKernel);
         camada.editarKernel(kernel);

         if(camada.temBias()){
            double[] bias = camada.bias();
            double[] gradB = camada.gradBias();
            idBias = calcular(bias, gradB, acb, idBias);
            camada.editarBias(bias);
         }
      }
   }

   /**
    * Atualiza as variáveis usando o gradiente pré calculado.
    * @param vars variáveis que serão atualizadas.
    * @param grads gradientes das variáveis.
    * @param acumulador acumulador do otimizador.
    * @param id índice inicial das variáveis dentro do array de momentums.
    * @return índice final após as atualizações.
    */
   private int calcular(double[] vars, double[] grads, double[] acumulador, int id){
      for(int i = 0; i < vars.length; i++){
         acumulador[id] = (rho * ac[id]) + ((1- rho) * (grads[i]*grads[i]));
         vars[i] -= (grads[i] * taxaAprendizagem) / (Math.sqrt(ac[id]) + epsilon);
         id++;
      }

      return id;
   }

   @Override
   public String info(){
      super.verificarConstrucao();
      super.construirInfo();
      
      super.addInfo("TaxaAprendizagem: " + this.taxaAprendizagem);
      super.addInfo("Rho: " + this.rho);
      super.addInfo("Epsilon: " + this.epsilon);

      return super.info();
   }

}
