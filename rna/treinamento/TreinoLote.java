package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.core.OpMatriz;
import rna.estrutura.CamadaDensa;
import rna.estrutura.RedeNeural;
import rna.otimizadores.Otimizador;

public class TreinoLote{
   OpMatriz mat = new OpMatriz();
   Auxiliar aux = new Auxiliar();
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
      CamadaDensa[] camadas = rede.obterCamadas();
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

   void backpropagationLote(CamadaDensa[] redec, Perda perda, double[] real){
      aux.calcularGradientes(redec, perda, real);

      for(CamadaDensa camada : redec){
         mat.add(camada.gradienteAcPesos, camada.gradientePesos, camada.gradienteAcPesos);
         mat.add(camada.gradienteAcBias, camada.gradienteBias, camada.gradienteAcBias);
      }
   }

   void zerarGradientesAcumulados(CamadaDensa[] redec){
      for(CamadaDensa camada : redec){
         mat.preencher(camada.gradienteAcPesos, 0);
         mat.preencher(camada.gradienteAcBias, 0);
      }
   }
   
   void calcularMediaGradientesLote(CamadaDensa[] redec, int tamLote){
      for(CamadaDensa camada : redec){
         
         for(int i = 0; i < camada.pesos.lin; i++){
            for(int j = 0; j < camada.pesos.col; j++){
               camada.gradienteAcPesos.div(i, j, tamLote);
            }
         }
         mat.copiar(camada.gradienteAcPesos, camada.gradientePesos);

         if(camada.temBias()){
            for(int i = 0; i < camada.bias.lin; i++){
               for(int j = 0; j < camada.bias.col; j++){
                  camada.gradienteAcBias.div(i, j, tamLote);
               }
            }
            mat.copiar(camada.gradienteAcBias, camada.gradienteBias);
         }

      }
   }
}
