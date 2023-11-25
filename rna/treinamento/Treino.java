package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.core.Mat;
import rna.core.Matriz;
import rna.estrutura.CamadaDensa;
import rna.estrutura.RedeNeural;
import rna.otimizadores.Otimizador;

public class Treino{
   Matriz mat = new Matriz();
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
      double[] dadoEntrada = new double[entrada[0].length];
      double[] dadoSaida = new double[saida[0].length];

      CamadaDensa[] camadas = rede.obterCamadas();
      Otimizador otimizador = rede.obterOtimizador();
      Perda perda = rede.obterPerda();
      
      double perdaEpoca;
      for(int e = 0; e < epochs; e++){
         aux.embaralharDados(entrada, saida);
         perdaEpoca = 0;
         
         for(int i = 0; i < entrada.length; i++){
            System.arraycopy(entrada[i], 0, dadoEntrada, 0, dadoEntrada.length);
            System.arraycopy(saida[i], 0, dadoSaida, 0, dadoSaida.length);

            rede.calcularSaida(dadoEntrada);

            //feedback de avanço da rede
            if(calcularHistorico){
               perdaEpoca += rede.obterPerda().calcular(rede.obterSaidas(), saida[i]);
            }

            backpropagation(camadas, perda, dadoSaida);  
            otimizador.atualizar(camadas);
         }

         //feedback de avanço da rede
         if(calcularHistorico){
            this.historico = aux.adicionarPerda(this.historico, perdaEpoca/entrada.length);
         }
      }
   }

   /**
    * Realiza a retropropagação de erros dentro da Rede Neural e calcula os gradientes
    * de cada camada para a atualização de pesos.
    * @param camadas conjunto de camadas densas da Rede Neural.
    * @param perda função de perda configurada para a Rede Neural.
    * @param real saída real que será usada para calcular os erros e gradientes.
    */
   public void backpropagation(CamadaDensa[] camadas, Perda perda, double[] real){
      aux.calcularErros(camadas, perda, real);

      //gradientes ou deltas para os pesos
      for(CamadaDensa camada : camadas){
         Mat entradaT = mat.transpor(camada.entrada);
         mat.mult(entradaT, camada.erros, camada.gradientes);
      }
   }

   @Deprecated
   public void atualizarPesos(CamadaDensa[] camadas, double taxaAprendizagem){
      for(int i = 0; i < camadas.length; i++){
         CamadaDensa camada = camadas[i];

         mat.escalar(camada.gradientes, taxaAprendizagem, camada.gradientes);
         mat.add(camada.pesos, camada.gradientes, camada.pesos);

         if(camada.temBias()){
            mat.escalar(camada.erros, taxaAprendizagem, camada.erros);
            mat.add(camada.bias, camada.erros, camada.bias);
         }
      }
   }
  
}
