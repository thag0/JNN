package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.core.OpMatriz;
import rna.estrutura.Densa;
import rna.modelos.RedeNeural;
import rna.otimizadores.Otimizador;

public class TreinoLote{
   OpMatriz opmat = new OpMatriz();
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
    * Treina a rede neural calculando os erros dos neuronios, seus gradientes para cada peso e 
    * passando essas informações para o otimizador configurado ajustar os pesos.
    * @param rede instância da rede.
    * @param perda função de perda (ou custo) usada para calcular os erros da rede.
    * @param otimizador otimizador configurado da rede.
    * @param entradas dados de entrada para o treino.
    * @param saidas dados de saída correspondente as entradas para o treino.
    * @param epochs quantidade de épocas de treinamento.
    * @param embaralhar embaralhar dados de treino para cada época.
    * @param tamLote tamanho do lote.
    */
   public void treinar(RedeNeural rede, double[][] entradas, double[][] saidas, int epochs, int tamLote){      
      Densa[] camadas = rede.obterCamadas();
      Otimizador otimizador = rede.obterOtimizador();
      Perda perda = rede.obterPerda();

      double perdaEpoca;
      for(int i = 0; i < epochs; i++){
         aux.embaralharDados(entradas, saidas);
         perdaEpoca = 0;

         for(int j = 0; j < entradas.length; j += tamLote){
            int fimIndice = Math.min(j + tamLote, entradas.length);
            double[][] entradaLote = aux.obterSubMatriz(entradas, j, fimIndice);
            double[][] saidaLote = aux.obterSubMatriz(saidas, j, fimIndice);

            //reiniciar gradiente do lote
            zerarGradientesAcumulados(camadas);
            for(int k = 0; k < entradaLote.length; k++){
               double[] entrada = entradaLote[k];
               double[] saida = saidaLote[k];

               rede.calcularSaida(entrada);
               if(this.calcularHistorico){
                  perdaEpoca += perda.calcular(rede.obterSaidas(), saidaLote[k]);
               }

               backpropagationLote(camadas, perda, saida);
            }

            //normalizar gradientes para enviar pro otimizador
            calcularMediaGradientesLote(camadas, entradaLote.length);
            otimizador.atualizar(camadas);
         }

         //feedback de avanço da rede
         if(this.calcularHistorico){
            this.historico = aux.adicionarPerda(this.historico, (perdaEpoca/tamLote));
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
    * @param redec conjunto de camadas densas da Rede Neural.
    * @param perda função de perda configurada para a Rede Neural.
    * @param real saída real que será usada para calcular os erros e gradientes.
    */
   void backpropagationLote(Densa[] redec, Perda perda, double[] real){
      aux.backpropagation(redec, perda, real);

      for(Densa camada : redec){
         opmat.add(camada.gradAcPesos, camada.gradPesos, camada.gradAcPesos);
         opmat.add(camada.gradAcBias, camada.gradBias, camada.gradAcBias);
      }
   }

   /**
    * Zera todos os acumuladores de gradientes das camadas (para pesos e bias)
    * para iniciar o treinamento de um lote.
    * @param redec conjunto de camadas densas da Rede Neural.
    */
   void zerarGradientesAcumulados(Densa[] redec){
      for(Densa camada : redec){
         opmat.preencher(camada.gradPesos, 0);
         opmat.preencher(camada.gradBias, 0);
         opmat.preencher(camada.gradAcPesos, 0);
         opmat.preencher(camada.gradAcBias, 0);
      }
   }

   /**
    * 
    * @param redec conjunto de camadas densas da Rede Neural.
    * @param tamLote tamanho do lote que foi usado para calcular os acumuladores
    * de gradiente das camadas.
    */
   void calcularMediaGradientesLote(Densa[] redec, int tamLote){
      for(Densa camada : redec){
         
         for(int i = 0; i < camada.pesos.lin; i++){
            for(int j = 0; j < camada.pesos.col; j++){
               camada.gradAcPesos.div(i, j, tamLote);
            }
         }
         opmat.copiar(camada.gradAcPesos, camada.gradPesos);

         if(camada.temBias()){
            for(int i = 0; i < camada.bias.lin; i++){
               for(int j = 0; j < camada.bias.col; j++){
                  camada.gradAcBias.div(i, j, tamLote);
               }
            }
            opmat.copiar(camada.gradAcBias, camada.gradBias);
         }

      }
   }
}
