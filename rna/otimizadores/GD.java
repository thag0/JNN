package rna.otimizadores;

import rna.core.OpArray;
import rna.estrutura.Camada;

/**
 * Classe que implementa o algoritmo de Descida do Gradiente para otimização de redes neurais.
 * Atualiza diretamente os pesos da rede com base no gradiente.
 * <p>
 *    O Gradiente descendente funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v[i][j] -= g[i][j] * tA
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - variável que será otimizadada (kernel, bias).
 * </p>
 *    {@code g} - gradiente correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - taxa de aprendizagem do otimizador.
 * </p>
 */
public class GD extends Otimizador{

   /**
    * Operador de arrays.
    */
   OpArray opArr = new OpArray();

   /**
    * Valor de taxa de aprendizagem do otimizador.
    */
   private double taxaAprendizagem;

   /**
    * Inicializa uma nova instância de otimizador da <strong> Descida do Gradiente </strong>
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA taxa de aprendizagem do otimizador.
    */
   public GD(double tA){
      this.taxaAprendizagem = tA;
   }

   /**
    * Inicializa uma nova instância de otimizador da <strong> Descida do Gradiente </strong>.
    * <p>
    *    Os hiperparâmetros do GD serão inicializados com os valores padrão, que são:
    * </p>
    * {@code taxaAprendizagem = 0.01}
    */
   public GD(){
      this(0.1);
   }

   @Override
   public void construir(Camada[] camadas){
      //esse otimizador não precisa de parâmetros adicionais
      this.construido = true;//otimizador pode ser usado
   }

   @Override
   public void atualizar(Camada[] camadas){
      super.verificarConstrucao();
      for(Camada camada : camadas){
         if(camada.treinavel == false) continue;

         double[] kernel = camada.obterKernel();
         double[] gradK = camada.obterGradKernel();
         
         for(int i = 0; i < kernel.length; i++){
            kernel[i] += gradK[i] * taxaAprendizagem;
         }
         camada.editarKernel(kernel);

         if(camada.temBias()){
            double[] bias = camada.obterBias();
            double[] gradB = camada.obterGradBias();
            
            for(int i = 0; i < bias.length; i++){
               bias[i] += gradB[i] * taxaAprendizagem;
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

      return super.info();
   }
   
}
