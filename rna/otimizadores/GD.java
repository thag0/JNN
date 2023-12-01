package rna.otimizadores;

import rna.core.OpMatriz;
import rna.estrutura.Densa;

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
 *    {@code v} - variável que será otimizadada (peso ou bias).
 * </p>
 *    {@code g} - gradiente correspondente a variável que será otimizada.
 * </p>
 * <p>
 *    {@code tA} - taxa de aprendizagem do otimizador.
 * </p>
 */
public class GD extends Otimizador{

   /**
    * Operador matricial para o otimizador.
    */
   OpMatriz mat = new OpMatriz();

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
      this(0.01);
   }

   @Override
   public void inicializar(Densa[] redec){
      
   }

   @Override
   public void atualizar(Densa[] redec){
      for(Densa camada : redec){
         mat.escalar(camada.gradPesos, taxaAprendizagem, camada.gradPesos);
         mat.add(camada.pesos, camada.gradPesos, camada.pesos);

         if(camada.temBias()){
            mat.escalar(camada.gradBias, taxaAprendizagem, camada.gradBias);
            mat.add(camada.bias, camada.gradBias, camada.bias);
         }
      } 
   }

   @Override
   public String info(){
      String buffer = "";

      String espacamento = "    ";
      buffer += espacamento + "TaxaAprendizagem: " + this.taxaAprendizagem + "\n";

      return buffer;
   }
   
}
