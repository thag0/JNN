package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.core.OpMatriz;
import rna.estrutura.Densa;
import rna.modelos.RedeNeural;
import rna.otimizadores.Otimizador;

public class Treino{
   OpMatriz opmat = new OpMatriz();
   Auxiliar aux = new Auxiliar();
   Random random = new Random();

   public boolean calcularHistorico = false;
   double[] historico;
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

      Densa[] camadas = rede.obterCamadas();
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
    * <p>
    *    Os gradientes iniciais são calculados usando a derivada da função de perda, com eles
    *    calculados, são retropropagados da última a primeira camada da rede.
    * </p>
    * @param redec conjunto de camadas densas da Rede Neural.
    * @param perda função de perda configurada para a Rede Neural.
    * @param real saída real que será usada para calcular os erros e gradientes.
    */
   public void backpropagation(Densa[] redec, Perda perda, double[] real){
      aux.backpropagation(redec, perda, real);
   }

   /**
    * Atualiza os gradientes diretanmente usando o gradiente descendente.
    * @param redec conjunto de camadas densas da Rede Neural.
    * @param taxaAprendizagem taxa de aprendizagem, que será aplicada ao gradientes
    * para as atualizações.
    */
   @Deprecated
   public void atualizarPesos(Densa[] redec, double taxaAprendizagem){
      for(int i = 0; i < redec.length; i++){
         Densa camada = redec[i];

         opmat.escalar(camada.gradPesos, taxaAprendizagem, camada.gradPesos);
         opmat.add(camada.pesos, camada.gradPesos, camada.pesos);

         if(camada.temBias()){
            opmat.escalar(camada.gradSaida, taxaAprendizagem, camada.gradSaida);
            opmat.add(camada.bias, camada.gradSaida, camada.bias);
         }
      }
   }
  
}
