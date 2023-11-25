package rna.treinamento;

import rna.estrutura.RedeNeural;

/**
 * Disponibilzia uma interface para usar os métodos de treino e treino em
 * lote da Rede Neural.
 */
public class Treinador{

   /**
    * Auxiliar na verificação do cálculo do histórico de custos.
    */
   public boolean calcularHistorico = false;

   Treino treino;
   TreinoLote treinoLote;

   /**
    * Responsável por organizar os tipos de treino da rede neural.
    */
   public Treinador(){
      treino =     new Treino(calcularHistorico);
      treinoLote = new TreinoLote(calcularHistorico);
   }

   /**
    * Configura a seed inicial do gerador de números aleatórios.
    * @param seed nova seed.
    */
   public void configurarSeed(long seed){
      this.treino.configurarSeed(seed);
      this.treinoLote.configurarSeed(seed);
   }

   /**
    * Configura o cálculo do custo da rede neural durante o processo de treinamento.
    * A mesma configuração se aplica ao treino em lote.
    * @param calcularHistorico calcular ou não o histórico de custo.
    */
    public void configurarHistoricoCusto(boolean calcularHistorico){
      this.calcularHistorico = calcularHistorico;
      treino.configurarHistorico(calcularHistorico);
   }

   /**
    * Treina a rede neural calculando os erros dos neuronios, seus gradientes para cada peso e 
    * passando essas informações para o otimizador configurado ajustar os pesos.
    * @param rede rede neural que será treinada.
    * @param perda função de perda usada durante o treinamento.
    * @param otimizador otimizador configurado da rede.
    * @param entradas dados de entrada para o treino.
    * @param saidas dados de saída correspondente as entradas para o treino.
    * @param epochs quantidade de épocas de treinamento.
    * @param embaralhar embaralhar dados de treino para cada época.
    */
   public void treino(RedeNeural rede, double[][] entradas, double[][] saidas, int epochs){
      treino.treinar(
         rede,
         clonarElementos(entradas), 
         clonarElementos(saidas), 
         epochs
      );

      treino.ultimoUsado = true;
      treinoLote.ultimoUsado = false;
   }

   /**
    * Treina a rede neural calculando os erros dos neuronios, seus gradientes para cada peso e 
    * passando essas informações para o otimizador configurado ajustar os pesos.
    * @param rede rede neural que será treinada.
    * @param perda função de perda usada durante o treinamento.
    * @param otimizador otimizador configurado da rede.
    * @param entradas dados de entrada para o treino.
    * @param saidas dados de saída correspondente as entradas para o treino.
    * @param epochs quantidade de épocas de treinamento.
    * @param embaralhar embaralhar dados de treino para cada época.
    * @param tamLote tamanho do lote.
    */
   public void treino(RedeNeural rede, double[][] entradas, double[][] saidas, int epochs, int tamLote){
      treinoLote.treinar(
         rede,
         clonarElementos(entradas), 
         clonarElementos(saidas), 
         epochs, 
         tamLote
      );

      treinoLote.ultimoUsado = true;
      treino.ultimoUsado = false;
   }

   /**
    * Copia elemento a elemento dos dados para evitar clones com mesmas referências
    * e embaralhar os dados de treino usados.
    * @param dados conjunto de dados base.
    * @return novo conjunto de dados com os mesmo valores contidos no original.
    */
   private double[][] clonarElementos(double[][] dados){
      double[][] clone = new double[dados.length][dados[0].length];

      for(int i = 0; i < dados.length; i++){
         // System.arraycopy(dados[i], 0, clone[i], 0, dados[i].length);
         for(int j = 0; j < dados[i].length; j++){
            clone[i][j] = dados[i][j];
         }
      }

      return clone;
   }

   /**
    * Retorna uma lista contendo os valores de custo da rede
    * a cada época de treinamento.
    * @return lista com os custo por época durante a fase de treinamento.
    */
   public double[] obterHistorico(){
      return treino.ultimoUsado ? treino.historico : treinoLote.historico;
   }
   
}
