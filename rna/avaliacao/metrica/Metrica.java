package rna.avaliacao.metrica;

import rna.estrutura.RedeNeural;


/**
 * Classe genérica para cálculos de métricas de avaliação da rede neural.
 * <p>
 *    Novas métricas devem implementar o método {@code calcular()}.
 * </p>
 */
abstract class Metrica{

   /**
    * Calcula a métrica de avaliação configurada.
    * @param rede rede neural.
    * @param entrada dados de entrada.
    * @param saida dados de saída relativos a entrada.
    * @return valor de avaliação de acordo com a métrica configurada
    */
   public double calcular(RedeNeural rede, double[][] entrada, double[][] saida){
      throw new UnsupportedOperationException(
         "É necessário implementar a métrica de avaliação da rede."
      );
   }


   /**
    * Calcula a métrica de avaliação configurada.
    * @param rede rede neural.
    * @param entrada dados de entrada.
    * @param saida dados de saída relativos a entrada.
    * @return valor de avaliação de acordo com a métrica configurada
    */
   public int[][] calcularMatriz(RedeNeural rede, double[][] entrada, double[][] saida){
      throw new UnsupportedOperationException(
         "É necessário implementar a métrica de avaliação da rede."
      );
   }


   /**
    * <p>
    *    Auxiliar.
    * </p>
    * Encontra o índice com o maior valor contido no array fornecido
    * @param dados array contendo os dados
    * @return índice com o maior valor contido nos dados.
    */
   protected int indiceMaiorValor(double[] dados){
      int indiceMaiorValor = 0;
      double maiorValor = dados[0];
  
      for(int i = 1; i < dados.length; i++){
         if (dados[i] > maiorValor) {
            maiorValor = dados[i];
            indiceMaiorValor = i;
         }
      }
  
      return indiceMaiorValor;
   }

   /**
   /**
    * <p>
    *    Auxiliar.
    * </p>
    * Calcula a matriz de confusão.
    * @param rede
    * @param entradas
    * @param saidas
    * @return
    */
   protected int[][] matrizConfusao(RedeNeural rede, double[][] entradas, double[][] saidas){
      int nClasses = saidas[0].length;
      int[][] matriz = new int[nClasses][nClasses];

      double[] entrada = new double[entradas[0].length];
      double[] saida = new double[saidas[0].length];
      double[] saidaRede = new double[rede.obterCamadaSaida().quantidadeNeuronios()];

      for(int i = 0; i < entradas.length; i++){
         System.arraycopy(entradas[i], 0, entrada, 0, entradas[i].length);
         System.arraycopy(saidas[i], 0, saida, 0, saidas[i].length);

         rede.calcularSaida(entrada);
         saidaRede = rede.obterSaidas();

         int real = this.indiceMaiorValor(saida);
         int previsto = this.indiceMaiorValor(saidaRede);

         matriz[real][previsto]++;
      }

      return matriz;
   } 
}
