package rna.treinamento;

import java.util.ArrayList;
import java.util.Random;
import rna.avaliacao.perda.Perda;
import rna.camadas.Camada;
import rna.core.Utils;
import rna.modelos.Modelo;
import rna.otimizadores.Otimizador;

class Treino {
   AuxTreino aux = new AuxTreino();
   Utils utils = new Utils();
   Random random = new Random();

   private boolean calcularHistorico = false;
   private ArrayList<Double> historico;
   boolean ultimoUsado = false;

   /**
    * Objeto de treino sequencial da rede.
    * @param historico lista de custos da rede durante cada época de treino.
    */
   public Treino(boolean calcularHistorico) {
      historico = new ArrayList<>(0);
      this.calcularHistorico = calcularHistorico;
   }

   /**
    * Configura a seed inicial do gerador de números aleatórios.
    * @param seed nova seed.
    */
   public void setSeed(long seed) {
      random.setSeed(seed);
      aux.setSeed(seed);
   }

   /**
    * Configura o cálculo de custos da rede neural durante cada
    * época de treinamento.
    * @param calcular caso verdadeiro, armazena os valores de custo da rede.
    */
   public void setHistorico(boolean calcular) {
      calcularHistorico = calcular;
   }

   /**
    * Treina a rede neural calculando os erros dos neuronios, seus gradientes para cada peso e 
    * passando essas informações para o otimizador configurado ajustar os pesos.
    * @param modelo instância da rede.
    * @param entrada dados de entrada para o treino.
    * @param saida dados de saída correspondente as entradas para o treino.
    * @param epochs quantidade de épocas de treinamento.
    * @param logs logs para perda durante as épocas de treinamento.
    */
   public void treinar(Modelo modelo, Object entrada, Object[] saida, int epochs, boolean logs) {
      Camada[] camadas = modelo.camadas();
      Otimizador otimizador = modelo.otimizador();
      Perda perda = modelo.perda();
      
      Object[] amostras = utils.transformarParaArray(entrada);
      Object[] rotulos = utils.transformarParaArray(saida);
      int numAmostras = amostras.length;

      if (logs) aux.esconderCursor();
      double perdaEpoca;
      for (int e = 1; e <= epochs; e++) {
         aux.embaralharDados(amostras, rotulos);
         perdaEpoca = 0;
         
         for (int i = 0; i < numAmostras; i++) {
            double[] amostraSaida = (double[]) rotulos[i];
            modelo.forward(amostras[i]);
            
            //feedback de avanço da rede
            if (calcularHistorico) {
               perdaEpoca += perda.calcular(modelo.saidaParaArray(), amostraSaida);
            }
            
            modelo.zerarGradientes();
            aux.backpropagation(camadas, perda, modelo.saidaParaArray(), amostraSaida);
            otimizador.atualizar(camadas);
         }

         if (logs) {
            aux.limparLinha();
            aux.exibirLogTreino("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca/numAmostras));
         }

         //feedback de avanço da rede
         if (calcularHistorico) historico.add(perdaEpoca/numAmostras);
      }

      if (logs) {
         aux.exibirCursor();
         System.out.println();
      } 
   }

   /**
    * Retorna o histórico de treino.
    * @return histórico de treino.
    */
   public Object[] historico(){
      return historico.toArray();
   }
  
}
