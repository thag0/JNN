package rna.treinamento;

import java.util.Random;
import rna.avaliacao.perda.Perda;
import rna.camadas.Camada;
import rna.modelos.Modelo;
import rna.otimizadores.Otimizador;

class Treino{
   AuxiliarTreino aux = new AuxiliarTreino();
   Random random = new Random();

   private boolean calcularHistorico = false;
   private double[] historico;
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
    * @param calcular caso verdadeiro, armazena os valores de custo da rede.
    */
   public void configurarHistorico(boolean calcular){
      this.calcularHistorico = calcular;
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
   public void treinar(Modelo modelo, Object[] entrada, Object[] saida, int epochs, boolean logs){
      Camada[] camadas = modelo.camadas();
      Otimizador otimizador = modelo.otimizador();
      Perda perda = modelo.perda();
      
      double perdaEpoca;
      for(int e = 1; e <= epochs; e++){
         aux.embaralharDados(entrada, saida);
         perdaEpoca = 0;
         
         for(int i = 0; i < entrada.length; i++){
            double[] amostraSaida = (double[]) saida[i];
            modelo.calcularSaida(entrada[i]);
            
            //feedback de avanço da rede
            if(calcularHistorico){
               perdaEpoca += perda.calcular(modelo.saidaParaArray(), amostraSaida);
            }
            
            backpropagation(camadas, perda, amostraSaida);
            otimizador.atualizar(camadas);
         }

         if(logs && (e % 5 == 0)){
            System.out.println("Época " +  e + "/" + epochs + " -> perda: " + (double)(perdaEpoca/entrada.length));
         }

         //feedback de avanço da rede
         if(calcularHistorico){
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
    * @param camadas conjunto de camadas de um modelo.
    * @param perda função de perda configurada para a Rede Neural.
    * @param real saída real que será usada para calcular os erros e gradientes.
    */
   public void backpropagation(Camada[] camadas, Perda perda, double[] real){
      aux.backpropagation(camadas, perda, real);
   }

   /**
    * Retorna o histórico de treino.
    * @return histórico de treino.
    */
   public double[] historico(){
      return this.historico;
   }
  
}
