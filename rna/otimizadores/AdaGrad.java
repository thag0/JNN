package rna.otimizadores;

import rna.core.OpArray;
import rna.estrutura.Camada;

/**
 * Implementa uma versão do algoritmo AdaGrad (Adaptive Gradient Algorithm).
 * O algoritmo otimiza o processo de aprendizado adaptando a taxa de aprendizagem 
 * de cada parâmetro com base no histórico de atualizações 
 * anteriores.
 * <p>
 *    Devido a natureza do otimizador, pode ser mais vantajoso (para este caso específico)
 *    usar valores de taxa de aprendizagem mais altos.
 * </p>
 * <p>
 *    O Adagrad funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v[i][j] -= (tA * g[i][j]) / (√ ac[i][j] + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizada (kernel, bias).
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

   /**
    * Valor padrão para a taxa de aprendizagem do otimizador.
    */
   private static final double PADRAO_TA = 0.99;

   /**
    * Valor padrão para o valor de epsilon pro otimizador.
    */
   private static final double PADRAO_EPS = 1e-7; 

   /**
    * Operador de arrays.
    */
   OpArray opArr = new OpArray();

   /**
    * Valor de taxa de aprendizagem do otimizador.
    */
   private double taxaAprendizagem;

   /**
    * Usado para evitar divisão por zero.
    */
   private double epsilon;

   /**
    * Acumuladores para os kernels.
    */
   private double[] ac;

   /**
    * Acumuladores para os bias.
    */
   private double[] acb;

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
      this(tA, PADRAO_EPS);
   }

   /**
    * Inicializa uma nova instância de otimizador <strong> AdaGrad </strong>.
    * <p>
    *    Os hiperparâmetros do AdaGrad serão inicializados com os valores padrão.
    * </p>
    */
   public AdaGrad(){
      this(PADRAO_TA, PADRAO_EPS);
   }

   @Override
   public void construir(Camada[] camadas){
      int nKernel = 0;
      int nBias = 0;
      
      for(Camada camada : camadas){
         if(camada.treinavel == false) continue;

         nKernel += camada.obterKernel().length;
         if(camada.temBias()){
            nBias += camada.obterBias().length;
         }         
      }

      this.ac  = new double[nKernel];
      this.acb = new double[nBias];
      double valorInicial = 0.1;

      opArr.preencher(ac, valorInicial);
      opArr.preencher(acb, valorInicial);
      this.construido = true;//otimizador pode ser usado
   }

   @Override
   public void atualizar(Camada[] camadas){
      super.verificarConstrucao();
      int i, idKernel = 0, idBias = 0;
      double g;

      for(Camada camada : camadas){
         if(camada.treinavel == false) continue;

         double[] kernel = camada.obterKernel();
         double[] gradP = camada.obterGradKernel();

         for(i = 0; i < kernel.length; i++){
            g = gradP[i];
            ac[idKernel] += g*g;
            kernel[i] += (g * taxaAprendizagem) / (Math.sqrt(ac[idKernel] + epsilon));
            idKernel++;
         }
         camada.editarKernel(kernel);
         
         if(camada.temBias()){
            double[] bias = camada.obterBias();
            double[] gradB = camada.obterGradBias();

            for(i = 0; i < bias.length; i++){
               g = gradB[i];
               acb[idBias] += g*g;
               bias[i] += (g * taxaAprendizagem) / (Math.sqrt(acb[idBias] + epsilon));
               idBias++;
            }
            camada.editarBias(bias);
         }
      }
   }

   @Override
   public String info(){
      super.verificarConstrucao();
      super.construirInfo();
      
      super.addInfo("TaxaAprendizagem: " + this.taxaAprendizagem);
      super.addInfo("Epsilon: " + this.epsilon);

      return super.info();
   }
}
