package jnn.otimizadores;

import jnn.camadas.Camada;

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
 *    v -= delta
 * </pre>
 * Onde delta é dado por:
 * <pre>
 * delta = √(acAt + eps) / √(ac + eps) * g
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada.
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
 *ac   = (rho * ac)   + ((1 - rho) * g²)
 *acAt = (rho * acAt) + ((1 - rho) * delta²)
 * </pre>
 * Onde:
 * <p>
 *    {@code rho} - constante de decaimento do otimizador.
 * </p>
 */
public class Adadelta extends Otimizador {

   /**
    * Valor padrão para a taxa de decaimento.
    */
   private static final double PADRAO_RHO = 0.999;

   /**
    * Valor padrão para epsilon.
    */
   private static final double PADRAO_EPS = 1e-6;

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
    * Acumulador atualizado para os kernels.
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
   public Adadelta(double rho, double epsilon) {
      if (rho <= 0) {
         throw new IllegalArgumentException(
            "\nTaxa de decaimento (" + rho + "), inválida."
         );
      }

      if (epsilon <= 0) {
         throw new IllegalArgumentException(
            "\nEpsilon (" + epsilon + "), inválido."
         );
      }

      this.rho = rho;
      this.epsilon = epsilon;
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Adadelta </strong> 
    * usando os valores de hiperparâmetros fornecidos.
    * @param rho valor de decaimento do otimizador.
    * @param epsilon usado para evitar a divisão por zero.
    */
   public Adadelta(double rho) {
      this(rho, PADRAO_EPS);
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> Adadelta </strong>.
    * <p>
    *    Os hiperparâmetros do Adadelta serão inicializados com os valores padrão.
    * </p>
    */
   public Adadelta() {
      this(PADRAO_RHO, PADRAO_EPS);
   }

   @Override
   public void construir(Camada[] camadas) {
      int nKernel = 0;
      int nBias = 0;
      
      for (Camada camada : camadas) {
         if (!camada.treinavel()) continue;

         nKernel += camada.kernelParaArray().length;
         if (camada.temBias()) {
            nBias += camada.biasParaArray().length;
         }         
      }

      this.ac  = new double[nKernel];
      this.acAt  = new double[nKernel];
      this.acb = new double[nBias];
      this.acAtb = new double[nBias];
      this._construido = true;//otimizador pode ser usado
   }

   @Override
   public void atualizar(Camada[] camadas) {
      verificarConstrucao();
      
      int idKernel = 0, idBias = 0;
      for (Camada camada : camadas) {
         if (!camada.treinavel()) continue;

         double[] kernel = camada.kernelParaArray();
         double[] gradK = camada.gradKernelParaArray();
         idKernel = calcular(kernel, gradK, ac, acAt, idKernel);
         camada.setKernel(kernel);

         if (camada.temBias()) {
            double[] bias = camada.biasParaArray();
            double[] gradB = camada.gradBias();
            idBias = calcular(bias, gradB, acb, acAtb, idBias);
            camada.setBias(bias);
         }
      }
   }

   /**
    * Atualiza as variáveis usando o gradiente pré calculado.
    * @param vars variáveis que serão atualizadas.
    * @param grads gradientes das variáveis.
    * @param ac acumulador do otimizador.
    * @param acAt acumulador atualizado.
    * @param id índice inicial das variáveis dentro do array de momentums.
    * @return índice final após as atualizações.
    */
   private int calcular(double[] vars, double[] grads, double[] ac, double[] acAt, int id) {
      double g, delta;

      for (int i = 0; i < vars.length; i++) {
         g = grads[i];
         ac[id] = (rho * ac[id]) + ((1 - rho) * (g*g));
         delta = Math.sqrt(acAt[id] + epsilon) / Math.sqrt(ac[id] + epsilon) * g;
         acAt[id] = (rho * acAt[id]) + ((1 - rho) * (delta * delta));
         vars[i] -= delta;
         
         id++;
      }

      return id;
   }

   @Override
   public String info() {
      super.verificarConstrucao();
      super.construirInfo();

      super.addInfo("Rho: " + this.rho);
      super.addInfo("Epsilon: " + this.epsilon);

      return super.info();
   }
}
