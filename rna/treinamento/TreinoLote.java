package rna.treinamento;

import java.util.ArrayList;
import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.camadas.Camada;
import rna.core.OpMatriz;
import rna.core.Utils;
import rna.core.OpArray;
import rna.modelos.Modelo;
import rna.otimizadores.Otimizador;

/**
 * Em testes ainda.
 */
class TreinoLote{
   OpMatriz opmat = new OpMatriz();
   OpArray oparr = new OpArray();
   Utils utils = new Utils();
   AuxiliarTreino aux = new AuxiliarTreino();
   Random random = new Random();

   public boolean calcularHistorico = false;
   private ArrayList<Double> historico;
   boolean ultimoUsado = false;

   /**
    * Implementação do treino em lote.
    * @param historico
    */
   public TreinoLote(boolean calcularHistorico){
      this.historico = new ArrayList<>(0);
      this.calcularHistorico = calcularHistorico;
   }

   /**
    * Configura a seed inicial do gerador de números aleatórios.
    * @param seed nova seed.
    */
    public void configurarSeed(long seed){
      this.random.setSeed(seed);
      this.aux.configurarSeed(seed);
   }

   /**
    * Configura o cálculo de custos da rede neural durante cada
    * época de treinamento.
    * @param calcularHistorico true armazena os valores de custo da rede, false não faz nada.
    */
    public void configurarHistorico(boolean calcularHistorico){
      this.calcularHistorico = calcularHistorico;
   }

   /**
    * Treina o modelo por um número determinado de épocas usando o treinamento em lotes.
    * @param modelo instância de modelo.
    * @param perda função de perda (ou custo) usada para calcular os erros da rede.
    * @param otimizador otimizador configurado do modelo.
    * @param entradas dados de entrada para o treino.
    * @param saidas dados de saída correspondente as entradas para o treino.
    * @param epochs quantidade de épocas de treinamento.
    * @param embaralhar embaralhar dados de treino para cada época.
    * @param tamLote tamanho do lote.
    * @param logs logs para perda durante as épocas de treinamento.
    */
   public void treinar(Modelo modelo, Object entradas, Object[] saidas, int epochs, int tamLote, boolean logs){
      Camada[] camadas = modelo.camadas();
      Otimizador otimizador = modelo.otimizador();
      Perda perda = modelo.perda();

      Object[] amostras = utils.transformarParaArray(entradas);
      int numAmostras = amostras.length;

      double perdaEpoca;
      for(int e = 1; e <= epochs; e++){
         aux.embaralharDados(amostras, saidas);
         perdaEpoca = 0;

         for(int i = 0; i < numAmostras; i += tamLote){
            int fimIndice = Math.min(i + tamLote, numAmostras);
            Object[] entradaLote = aux.obterSubMatriz(amostras, i, fimIndice);
            Object[] saidaLote = aux.obterSubMatriz(saidas, i, fimIndice);
            
            modelo.zerarGradientes();//zerar gradientes para o acumular pelo lote
            for(int j = 0; j < entradaLote.length; j++){
               double[] saidaAmostra = (double[]) saidaLote[j];
               modelo.calcularSaida(entradaLote[j]);

               if(this.calcularHistorico){
                  perdaEpoca += perda.calcular(modelo.saidaParaArray(), saidaAmostra);
               }

               aux.backpropagation(camadas, perda, saidaAmostra);
            }

            otimizador.atualizar(camadas);
         }

         if(logs && (e % 5 == 0)){
            System.out.println("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca/numAmostras));
         }

         //feedback de avanço da rede
         if(calcularHistorico){
            historico.add((perdaEpoca/numAmostras));
         }
      }
   }

   /**
    * Retorna o histórico de treino.
    * @return histórico de treino.
    */
   public Object[] historico(){
      return this.historico.toArray();
   }
}
