package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.camadas.Camada;
import rna.core.OpMatriz;
import rna.core.OpArray;
import rna.modelos.Modelo;
import rna.otimizadores.Otimizador;

/**
 * Em testes ainda.
 */
public class TreinoLote{
   OpMatriz opmat = new OpMatriz();
   OpArray oparr = new OpArray();
   AuxiliarTreino aux = new AuxiliarTreino();
   Random random = new Random();

   public boolean calcularHistorico = false;
   double[] historico;
   boolean ultimoUsado = false;

   /**
    * Implementação do treino em lote.
    * @param historico
    */
   public TreinoLote(boolean calcularHistorico){
      this.historico = new double[0];
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
   public void treinar(Modelo modelo, Object[] entradas, Object[] saidas, int epochs, int tamLote, boolean logs){
      Camada[] camadas = modelo.camadas();
      Otimizador otimizador = modelo.otimizador();
      Perda perda = modelo.perda();

      double perdaEpoca;
      for(int e = 0; e < epochs; e++){
         aux.embaralharDados(entradas, saidas);
         perdaEpoca = 0;

         for(int i = 0; i < entradas.length; i += tamLote){
            int fimIndice = Math.min(i + tamLote, entradas.length);
            Object[] entradaLote = aux.obterSubMatriz(entradas, i, fimIndice);
            Object[] saidaLote = aux.obterSubMatriz(saidas, i, fimIndice);

            //reiniciar gradiente do lote
            zerarGradientesAcumulados(camadas);
            for(int j = 0; j < entradaLote.length; j++){
               double[] saidaAmostra = (double[]) saidaLote[j];

               modelo.calcularSaida(entradaLote[j]);
               if(this.calcularHistorico){
                  perdaEpoca += perda.calcular(modelo.saidaParaArray(), saidaAmostra);
               }

               backpropagationLote(camadas, perda, saidaAmostra);
            }

            //normalizar gradientes para enviar pro otimizador
            calcularMediaGradientesLote(camadas, entradaLote.length);
            otimizador.atualizar(camadas);
         }

         //feedback de avanço da rede
         if(this.calcularHistorico){
            this.historico = aux.addPerda(this.historico, (perdaEpoca/tamLote));
         }
      }
   }

   /**
    * Realiza a retropropagação de gradientes de cada camada para a atualização de pesos.
    * <p>
    *    Os gradientes iniciais são calculados usando a derivada da função de perda, com eles
    *    calculados, são retropropagados da última a primeira camada da rede.
    * </p>
    * Ao final os gradientes calculados são adicionados aos acumuladores para o lote.
    * @param camadas conjunto de camadas do modelo.
    * @param perda função de perda do modelo.
    * @param real saída real que será usada para calcular os erros e gradientes.
    */
   void backpropagationLote(Camada[] camadas, Perda perda, double[] real){
      aux.backpropagation(camadas, perda, real);

      for(Camada camada : camadas){
         if(camada.treinavel == false) continue;

         double[] gradK = camada.obterGradKernel();
         double[] acK   = camada.obterAcGradKernel();
         oparr.add(acK, gradK, acK);
         camada.editarAcGradKernel(acK);

         if(camada.temBias()){
            double[] gradB = camada.obterGradBias();
            double[] acB   = camada.obterAcGradBias();
            oparr.add(acB, gradB, acB);
            camada.editarAcGradBias(acB);
         }     
      }
   }

   /**
    * Zera todos os acumuladores de gradientes das camadas (para kernels e bias)
    * para iniciar o treinamento de um lote.
    * @param camadas conjunto de camadas do modelo.
    */
   void zerarGradientesAcumulados(Camada[] camadas){
      for(Camada camada : camadas){
         if(camada.treinavel == false) continue;
         camada.zerarAcumuladores();
      }
   }

   /**
    * 
    * @param redec conjunto de camadas do modelo.
    * @param tamLote tamanho do lote usado para o treino.
    */
   void calcularMediaGradientesLote(Camada[] redec, int tamLote){
      double tamanho = (double)tamLote;

      for(Camada camada : redec){
         if(camada.treinavel == false) continue;
         
         double[] acKernel = camada.obterAcGradKernel();
         oparr.dividirEscalar(acKernel, tamanho, acKernel);
         camada.editarGradienteKernel(acKernel);

         if(camada.temBias()){
            double[] acBias = camada.obterAcGradBias();
            oparr.dividirEscalar(acBias, tamanho, acBias);
            camada.editarGradienteBias(acBias);
         }
      }
   }
}
