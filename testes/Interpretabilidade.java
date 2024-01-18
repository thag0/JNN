package testes;

import lib.ged.Dados;
import lib.ged.Ged;
import rna.camadas.Camada;
import rna.camadas.Densa;
import rna.core.Mat;
import rna.inicializadores.Xavier;
import rna.modelos.Modelo;
import rna.modelos.Sequencial;

public class Interpretabilidade{
   static Ged ged = new Ged();

   public static void main(String[] args){
      ged.limparConsole();
      double[][] entrada = {
         {0, 0},
         {0, 1},
         {1, 0},
         {1, 1},
      };
      double[][] saida = {
         {0},
         {1},
         {1},
         {0}
      };
      int tamEntrada = entrada[0].length, tamSaida = saida[0].length;

      // Criação e treino do modelo
      Sequencial modelo = new Sequencial(new Camada[]{
         new Densa(tamEntrada, 2, "sigmoid"),
         new Densa(tamSaida, "sigmoid")
      });
      modelo.configurarSeed(222222222);
      modelo.compilar("adagrad", "mse", new Xavier());
      modelo.configurarHistorico(true);
      modelo.treinar(entrada, saida, 800, false);
      
      //avaliação de resultados
      System.out.println("Perda: " + modelo.avaliar(entrada, saida));
      verificar(modelo, entrada, saida);

      //gráfico de perda durante o treino
      String caminhoHistorico = "historico-perda";
      exportarHistorico(modelo, caminhoHistorico);
      new Thread(() -> {
         executarComando("python grafico.py " + caminhoHistorico);
      }).start();

      //"Interpretabilidade"
      Densa[] camadas = new Densa[modelo.numCamadas()];
      for(int i = 0; i < modelo.numCamadas(); i++){
         camadas[i] = (Densa) modelo.camada(i);
      }

      System.out.println();
      exibirParametros(camadas);

      double[][] pred = obterPredicoes(camadas, entrada);
      Mat neuronio1 = new Mat(pred[0]);
      neuronio1.configurarFormato(4, 1);
      Mat neuronio2 = new Mat(pred[1]);
      neuronio2.configurarFormato(4, 1);
      Mat neuronio3 = new Mat(pred[2]);
      neuronio3.configurarFormato(4, 1);

      neuronio1.print("neuronio 1", 4);
      neuronio2.print("neuronio 2", 4);
      neuronio3.print("neuronio 3", 4);
   }

   static double[][] obterPredicoes(Densa[] camadas, double[][] entrada){
      double[][] pred = new double[3][entrada.length];

      for(int i = 0; i < entrada.length; i++){
         camadas[0].calcularSaida(entrada[i]);
         camadas[1].calcularSaida(camadas[0].saida());
      
         pred[0][i] = camadas[0].saidaParaArray()[0];
         pred[1][i] = camadas[0].saidaParaArray()[1];
         pred[2][i] = camadas[1].saidaParaArray()[0];
      }

      return pred;
   }

   static void exibirParametros(Densa[] camadas){
      for(int i = 0; i < camadas.length; i++){
         camadas[i].pesos.print("pesos c" + i, 4);
         camadas[i].bias.print("bias c" + i, 4);
         System.out.println();
      }
   }

   static void verificar(Sequencial modelo, double[][] entrada, double[][] saida){
      for(int i = 0; i < entrada.length; i++){
         modelo.calcularSaida(entrada[i]);
         System.out.println(
            entrada[i][0] + " - " + entrada[i][1] + 
            " R: " + saida[i][0] + 
            " P: " + modelo.saidaParaArray()[0]
         );
      }
   }

   static void exportarHistorico(Modelo modelo, String caminho){
      System.out.println("Exportando histórico de perda");
      double[] perdas = modelo.historico();
      double[][] dadosPerdas = new double[perdas.length][1];

      for(int i = 0; i < dadosPerdas.length; i++){
         dadosPerdas[i][0] = perdas[i];
      }

      Dados dados = new Dados(dadosPerdas);
      ged.exportarCsv(dados, caminho);
   }

   static void executarComando(String comando){
      try{
         new ProcessBuilder("cmd", "/c", comando).inheritIO().start().waitFor();
      }catch(Exception e){
         e.printStackTrace();
      }
   }
}
