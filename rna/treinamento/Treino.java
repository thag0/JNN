package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.camadas.Camada;
import rna.camadas.Densa;
import rna.core.OpMatriz;
import rna.modelos.Modelo;
import rna.otimizadores.Otimizador;

public class Treino{
   OpMatriz opmat = new OpMatriz();
   AuxiliarTreino aux = new AuxiliarTreino();
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
    * @param modelo instância da rede.
    * @param entrada dados de entrada para o treino.
    * @param saida dados de saída correspondente as entradas para o treino.
    * @param epochs quantidade de épocas de treinamento.
    */
   public void treinar(Modelo modelo, Object[] entrada, Object[] saida, int epochs, boolean logs){
      Camada[] camadas = modelo.camadas();
      Otimizador otimizador = modelo.otimizador();
      Perda perda = modelo.perda();
      
      double perdaEpoca;
      for(int e = 0; e < epochs; e++){
         aux.embaralharDados(entrada, saida);
         perdaEpoca = 0;
         
         for(int i = 0; i < entrada.length; i++){
            double[] amostraSaida = (double[]) saida[i];
            modelo.calcularSaida(entrada[i]);
            
            //feedback de avanço da rede
            if(this.calcularHistorico){
               perdaEpoca += perda.calcular(modelo.saidaParaArray(), amostraSaida);
            }
            
            backpropagation(camadas, perda, amostraSaida);  
            otimizador.atualizar(camadas);
         }

         if(logs & (e % 5 == 0)){
            System.out.println("Perda (" + e + "): " + (double)(perdaEpoca/entrada.length));
         }

         //feedback de avanço da rede
         if(this.calcularHistorico){
            this.historico = aux.addPerda(this.historico, (double)(perdaEpoca/entrada.length));
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
   public void backpropagation(Camada[] redec, Perda perda, double[] real){
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

         opmat.multEscalar(camada.gradPesos, taxaAprendizagem, camada.gradPesos);
         opmat.add(camada.pesos, camada.gradPesos, camada.pesos);

         if(camada.temBias()){
            opmat.multEscalar(camada.gradSaida, taxaAprendizagem, camada.gradSaida);
            opmat.add(camada.bias, camada.gradSaida, camada.bias);
         }
      }
   }
  
}
