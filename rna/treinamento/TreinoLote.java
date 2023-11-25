package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.core.Mat;
import rna.core.Matriz;
import rna.estrutura.CamadaDensa;
import rna.estrutura.RedeNeural;
import rna.otimizadores.Otimizador;

public class TreinoLote{
   Matriz mat = new Matriz();
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
               backpropagationLote(camadas, perda, saida);
            }

            if(this.calcularHistorico){
               perdaEpoca += perda.calcular(rede.obterSaidas(), saidaLote[i]);
            }

            //normalizar gradientes para enviar pro otimizador
            calcularMediaGradientesLote(camadas, entradaLote.length);
            otimizador.atualizar(camadas);
         }

         //feedback de avanço da rede
         if(calcularHistorico){
            this.historico = aux.adicionarPerda(this.historico, perdaEpoca/tamLote);
         }
      }
   }

   void backpropagationLote(CamadaDensa[] redec, Perda perda, double[] real){
      aux.calcularErros(redec, perda, real);

      //gradientes ou delta para os pesos
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];
         Mat entradaT = mat.transpor(camada.entrada);
         mat.mult(entradaT, camada.erros, camada.gradientes);
         mat.add(camada.gradientes, camada.gradientesAcumulados, camada.gradientesAcumulados);
      }
   }

   void zerarGradientesAcumulados(CamadaDensa[] redec){
      for(CamadaDensa camada : redec){
         mat.preencher(camada.gradientesAcumulados, 0);
      }
   }
   
   void calcularMediaGradientesLote(CamadaDensa[] redec, int tamLote){
      for(CamadaDensa camada : redec){
         mat.copiar(camada.gradientesAcumulados, camada.gradientes);
         
         for(int i = 0; i < camada.gradientes.lin; i++){
            for(int j = 0; j < camada.gradientes.col; j++){
               camada.gradientes.div(i, j, tamLote);
            }
         }
      }
   }
}
