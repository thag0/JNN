package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.core.OpMatriz;
import rna.estrutura.CamadaDensa;
import rna.estrutura.RedeNeural;
import rna.otimizadores.Otimizador;

public class Treino{
   OpMatriz mat = new OpMatriz();
   public boolean calcularHistorico = false;
   double[] historico;
   Auxiliar aux = new Auxiliar();

   Random random = new Random();
   boolean ultimoUsado = false;

   /**
    * Objeto de treino sequencial da rede.
    * @param historico lista de custos da rede durante cada época de treino.
    */
    public Treino(boolean calcularHistorico){
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
    * @param entrada dados de entrada para o treino.
    * @param saida dados de saída correspondente as entradas para o treino.
    * @param epochs quantidade de épocas de treinamento.
    */
   public void treinar(RedeNeural rede, double[][] entrada, double[][] saida, int epochs){
      double[] amostraEntrada = new double[entrada[0].length];
      double[] amostraSaida = new double[saida[0].length];

      CamadaDensa[] camadas = rede.obterCamadas();
      Otimizador otimizador = rede.obterOtimizador();
      Perda perda = rede.obterPerda();
      
      double perdaEpoca;
      for(int e = 0; e < epochs; e++){
         aux.embaralharDados(entrada, saida);
         perdaEpoca = 0;
         
         for(int i = 0; i < entrada.length; i++){
            aux.copiarArray(entrada[i], amostraEntrada);
            aux.copiarArray(saida[i], amostraSaida);

            rede.calcularSaida(amostraEntrada);

            //feedback de avanço da rede
            if(this.calcularHistorico){
               perdaEpoca += perda.calcular(rede.obterSaidas(), saida[i]);
            }

            backpropagation(camadas, perda, amostraSaida);  
            otimizador.atualizar(camadas);
         }

         //feedback de avanço da rede
         if(this.calcularHistorico){
            this.historico = aux.adicionarPerda(this.historico, perdaEpoca/entrada.length);
         }
      }
   }

   /**
    * Realiza a retropropagação de gradientes de cada camada para a atualização de pesos.
    * @param redec conjunto de camadas densas da Rede Neural.
    * @param perda função de perda configurada para a Rede Neural.
    * @param real saída real que será usada para calcular os erros e gradientes.
    */
   public void backpropagation(CamadaDensa[] redec, Perda perda, double[] real){
      CamadaDensa saida = redec[redec.length-1];
      double[] previsto = saida.obterSaida().linha(0);
      double[] gradSaida = perda.derivada(previsto, real);
      saida.calcularGradiente(gradSaida);

      for(int i = redec.length-2; i >= 0; i--){
         redec[i].calcularGradiente(redec[i+1].gradienteEntrada.linha(0));
      }
   }

   @Deprecated
   public void atualizarPesos(CamadaDensa[] camadas, double taxaAprendizagem){
      for(int i = 0; i < camadas.length; i++){
         CamadaDensa camada = camadas[i];

         mat.escalar(camada.gradientePesos, taxaAprendizagem, camada.gradientePesos);
         mat.add(camada.pesos, camada.gradientePesos, camada.pesos);

         if(camada.temBias()){
            mat.escalar(camada.gradienteSaida, taxaAprendizagem, camada.gradienteSaida);
            mat.add(camada.bias, camada.gradienteSaida, camada.bias);
         }
      }
   }
  
}
