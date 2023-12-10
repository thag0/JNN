package rna.treinamento;

import rna.modelos.Modelo;

/**
 * Disponibilzia uma interface para usar os métodos de treino e treino em
 * lote da Rede Neural.
 */
public class Treinador{

   /**
    * Auxiliar na verificação do cálculo do histórico de custos.
    */
   public boolean calcularHistorico = false;
   
   AuxiliarTreino aux;
   Treino treino;
   TreinoLote treinoLote;

   /**
    * Responsável por organizar os tipos de treino da rede neural.
    */
   public Treinador(){
      aux =        new AuxiliarTreino();
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
      treinoLote.configurarHistorico(calcularHistorico);
   }

   /**
    * Treina a rede neural calculando os erros dos neuronios, seus gradientes para cada peso e 
    * passando essas informações para o otimizador configurado ajustar os pesos.
    * @param modelo rede neural que será treinada.
    * @param entradas dados de entrada para o treino.
    * @param saidas dados de saída correspondente as entradas para o treino.
    * @param epochs quantidade de épocas de treinamento.
    */
   public void treino(Modelo modelo, Object[] entradas, Object[] saidas, int epochs){
      treino.treinar(
         modelo,
         entradas.clone(), 
         saidas.clone(), 
         epochs
      );

      treino.ultimoUsado = true;
      treinoLote.ultimoUsado = false;
   }

   /**
    * Treina a rede neural calculando os erros dos neuronios, seus gradientes para cada peso e 
    * passando essas informações para o otimizador configurado ajustar os pesos.
    * @param rede rede neural que será treinada.
    * @param entradas dados de entrada para o treino.
    * @param saidas dados de saída correspondente as entradas para o treino.
    * @param epochs quantidade de épocas de treinamento.
    * @param tamLote tamanho do lote.
    */
   public void treino(Modelo rede, Object[] entradas, Object[] saidas, int epochs, int tamLote){
      treinoLote.treinar(
         rede,
         entradas, 
         saidas, 
         epochs, 
         tamLote
      );

      treinoLote.ultimoUsado = true;
      treino.ultimoUsado = false;
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
