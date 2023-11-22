package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.core.Matriz;
import rna.estrutura.CamadaDensa;
import rna.estrutura.RedeNeural;

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

   public void treinar(RedeNeural rede, double[][] entrada, double[][] saida, int epochs){
      double[] dadoEntrada = new double[entrada[0].length];
      double[] dadoSaida = new double[saida[0].length];
      
      for(int e = 0; e < epochs; e++){
         aux.embaralharDados(entrada, saida);
         
         for(int i = 0; i < entrada.length; i++){
            System.arraycopy(entrada[i], 0, dadoEntrada, 0, dadoEntrada.length);
            System.arraycopy(saida[i], 0, dadoSaida, 0, dadoSaida.length);

            rede.calcularSaida(dadoEntrada);
            backpropagation(rede.camadas, rede.obterPerda(), dadoSaida);  
            
            // atualizarPesos(rede.camadas, 0.01);
            rede.obterOtimizador().atualizar(rede.obterCamadas());
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

      //gradientes ou delta para os pesos
      for(CamadaDensa camada : camadas){
         double[][] entradaT = mat.transpor(camada.entrada);
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
