package rna.avaliacao;

import rna.avaliacao.metrica.*;
import rna.avaliacao.perda.*;
import rna.core.Tensor4D;
import rna.modelos.Modelo;

/**
 * Contém implementações de métricas para analisar modelos.
 */
public class Avaliador {
   private Modelo modelo;

   EntropiaCruzada ecc = new EntropiaCruzada();
   EntropiaCruzadaBinaria ecb = new EntropiaCruzadaBinaria();
   Acuracia acuracia = new Acuracia();
   MatrizConfusao matrizConfusao = new MatrizConfusao();
   F1Score f1Score = new F1Score();
   MSE emq = new MSE();
   MAE ema = new MAE();
   MSLE emql = new MSLE();   

   /**
    * Instancia um novo avaliador destinado a uma Rede Neural.
    * @param modelo rede neural para o avaliador.
    * @throws IllegalArgumentException se a rede fornecida for nula.
    */
   public Avaliador(Modelo modelo) {
      if (modelo == null) {
         throw new IllegalArgumentException("\nO modelo fornecido não pode ser nulo.");
      }

      this.modelo = modelo;
   }

   /**
    * Transforma o conteúdo de saída em um array double[][].
    * @param saida array de saída desejada.
    * @return array no formato {@code double[][]}
    */
   private double[][] saidaParaArray(Object[] saida) {
      double[][] s;
      if (saida instanceof double[][]) {
         s = (double[][]) saida;
      
      }else if (saida instanceof Tensor4D[]) {
         Tensor4D[] arr = (Tensor4D[]) saida;
         s = new double[arr.length][];

         //por padrão usar só a primeira linha do tensor.
         for (int i = 0; i < arr.length; i++) {
            s[i] = arr[i].array1D(0, 0, 0);
         }
      
      } else {
         throw new IllegalArgumentException(
            "O formato de saída deve ser do tipo double[][] ou Tensor4D, recebido " +
            saida.getClass().getTypeName()
         );
      }

      return s;
   }

   /**
    * Calcula o erro médio quadrado da rede neural em relação aos dados previstos e reais.
    * @param previsto dados previstos.
    * @param real dados rotulados.
    * @return valor do erro médio quadrado da rede em relação ao dados fornecidos (custo/perda).
    */
   public double erroMedioQuadrado(double[] previsto, double[] real) {
      return ema.calcular(previsto, real);
   }

   /**
    * Calcula o erro médio quadrado médio da rede neural em relação aos dados de entrada e 
    * saída fornecidos.
    * @param entrada matriz com  os dados de entrada (features).
    * @param saida matriz com os dados de saída (classes).
    * @return valor do erro médio quadrado da rede em relação ao dados fornecidos (custo/perda).
    */
   public double erroMedioQuadrado(Object[] entrada, Object[] saida) {
      Tensor4D[] previsoes = modelo.forwards(entrada);
      double[][] s = saidaParaArray(saida);

      int tam = entrada.length;
      double res = 0;
      for (int i = 0; i < tam; i++) {
         res += ema.calcular(previsoes[i].array1D(0, 0, 0), s[i]);
      }

      return res / tam;
   }

   /**
    * Calcula o erro médio quadrado logarítimoco da rede neural em relação aos dados 
    * previstos e reais.
    * @param previsto dados previstos.
    * @param real dados rotulados.
    * @return valor do erro médio quadrado logarítimico da rede em relação ao dados 
    * fornecidos (custo/perda).
    */
   public double erroMedioQuadradoLogaritmico(double[] previsto, double[] real) {
      return emql.calcular(previsto, real);
   }

   /**
    * Calcula o erro médio quadrado logarítimico médio da rede neural em relação aos dados 
    * de entrada e saída fornecidos.
    * @param entrada matriz com  os dados de entrada (features).
    * @param saida matriz com os dados de saída (classes).
    * @return valor do erro médio quadrado logarítimico da rede em relação ao dados 
    * fornecidos (custo/perda).
    */
   public double erroMedioQuadradoLogaritmico(Object[] entrada, Object[] saida) {
      Tensor4D[] previsoes = modelo.forwards(entrada);
      double[][] s = saidaParaArray(saida);

      int tam = entrada.length;
      double res = 0;
      for (int i = 0; i < tam; i++) {
         res += emql.calcular(previsoes[i].array1D(0, 0, 0), s[i]);
      }

      return res/tam; 
   }

   /**
    * Calcula o erro médio absoluto da rede neural em relação aos dados previstos e reais.
    * @param previsto dados previstos.
    * @param real dados rotulados.
    * @return valor do erro médio abosoluto da rede em relação ao dados fornecidos (custo/perda).
    */
   public double erroMedioAbsoluto(double[] previsto, double[] real) {
      return ema.calcular(previsto, real);
   }

   /**
    * Calcula o erro médio absoluto entre as saídas previstas pela rede neural e os valores reais.
    * @param entrada dados de entrada.
    * @param saida dados de saída contendo os resultados respectivos para as entradas.
    * @return valor do erro médio abosoluto da rede em relação ao dados fornecidos (custo/perda).
    */
   public double erroMedioAbsoluto(Object[] entrada, Object[] saida) {
      Tensor4D[] previsoes = modelo.forwards(entrada);
      double[][] s = saidaParaArray(saida);

      int tam = entrada.length;
      double res = 0;
      for (int i = 0; i < tam; i++) {
         res += ema.calcular(previsoes[i].array1D(0, 0, 0), s[i]);
      }

      return res/tam;
   }

   /**
    * Calcula a precisão da rede neural em relação aos dados de entrada e saída fornecidos.
    * @param entrada dados de entrada.
    * @param saida dados de saída contendo os resultados respectivos para as entradas.
    * @return A acurácia da rede neural em forma de probabilidade.
    */
   public double acuracia(Object entrada, Object[] saida) {
      return acuracia.calcular(this.modelo, entrada, saida);
   }

   /**
    * Calcula a entropia cruzada da rede neural em relação aos dados previstos e reais.
    * e as saídas reais fornecidas.
    * @param previsto dados previstos.
    * @param real dados rotulados.
    * @return entropia cruzada da rede em relação ao dados fornecidos (custo/perda).
    */
   public double entropiaCruzada(double[] previsto, double[] real) {  
      return ecc.calcular(previsto, real);
   }

   /**
    * Calcula a entropia cruzada entre as saídas previstas pela rede neural
    * e as saídas reais fornecidas.
    * @param entrada dados de entrada.
    * @param saida dados de saída contendo os resultados respectivos para as entradas.
    * @return entropia cruzada da rede em relação ao dados fornecidos (custo/perda).
    */
   public double entropiaCruzada(Object[] entrada, Object[] saida) {  
      Tensor4D[] previsoes = modelo.forwards(entrada);
      double[][] s = saidaParaArray(saida);

      int tam = entrada.length;
      double res = 0;
      for (int i = 0; i < tam; i++) {
         res += ecc.calcular(previsoes[i].array1D(0, 0, 0), s[i]);
      }

      return res/tam;
   }

   /**
    * Calcula a entropia cruzada binária entre as saídas previstas pela rede neural
    * e as saídas reais fornecidas.
    * @param previsto dados previstos.
    * @param real dados rotulados.
    * @return valor da entropia cruzada binária da rede em relação ao dados fornecidos (custo/perda).
    */
   public double entropiaCruzadaBinaria(double[] previsto, double[] real) {
      return ecb.calcular(previsto, real);
   }

   /**
    * Calcula a entropia cruzada binária entre as saídas previstas pela rede neural
    * e as saídas reais fornecidas.
    * @param entrada Os dados de entrada para os quais a rede neural calculará as saídas.
    * @param saida As saídas reais correspondentes aos dados de entrada.
    * @return valor da entropia cruzada binária.
    */
   public double entropiaCruzadaBinaria(Object[] entrada, Object[] saida) {
      Tensor4D[] previsoes = modelo.forwards(entrada);
      double[][] s = saidaParaArray(saida);

      int tam = entrada.length;
      double res = 0;
      for (int i = 0; i < tam; i++) {
         res += ecb.calcular(previsoes[i].array1D(0, 0, 0), s[i]);
      }

      return res/tam;
   }

   /**
    * Calcula a matriz de confusão para avaliar o desempenho da rede em classificação.
    * <p>
    *    A matriz de confusão mostra a contagem de amostras que foram classificadas de forma 
    *    correta ou não em cada classe. As linhas representam as classes reais e as colunas as 
    *    classes previstas pela rede.
    * </p>
    * @param entradas matriz com os dados de entrada 
    * @param saidas matriz com os dados de saída
    * @return matriz de confusão para avaliar o desempenho do modelo.
    * @throws IllegalArgumentException se o modelo não foi compilado previamente.
    */
   public int[][] matrizConfusao(Object entradas, double[][] saidas) {
      return matrizConfusao.calcularMatriz(this.modelo, entradas, saidas);
   }

   /**
    * Calcula o F1-Score ponderado para o modelo de rede neural em relação às entradas e 
    * saídas fornecidas.
    * <p>
    *    O F1-Score é uma métrica que combina a precisão e o recall para avaliar o desempenho 
    *    de um modelo de classificação. Ele é especialmente útil quando se lida com classes 
    *    desbalanceadas ou quando se deseja equilibrar a precisão e o recall.
    * </p>
    * @param entradas matriz com os dados de entrada 
    * @param saidas matriz com os dados de saída
    * @return f1-score ponderado para o modelo em relação aos dados de entrada e saída.
    */
   public double f1Score(double[][] entradas, double[][] saidas) {
      return f1Score.calcular(this.modelo, entradas, saidas);
   }
}
