package testes;

import lib.ged.Dados;
import lib.ged.Ged;
import rna.camadas.*;
import rna.modelos.*;

public class TesteTreino{
   static Ged ged = new Ged();

   public static void main(String[] args){
      ged.limparConsole();

      double[][] entrada = {
         {0, 0},
         {0, 1},
         {1, 0},
         {1, 1}
      };
      double[][] saida = {
         {0},
         {1},
         {1},
         {0}
      };

      Sequencial modelo = new Sequencial(new Camada[]{
         new Densa(2, 3, "sigmoid"),
         new Densa(1, "sigmoid")
      });
      
      modelo.compilar("adagrad", "mse");
      modelo.treinar(entrada, saida, 3_000, false);
      verificar(entrada, saida, modelo);
   }

   static void verificar(double[][] entrada, double[][] saida, Sequencial modelo){
      for(int i = 0; i < entrada.length; i++){
         modelo.calcularSaida(entrada[i]);
         System.out.println(
            entrada[i][0] + " - " + entrada[i][1] + 
            " R: " + saida[i][0] + 
            " P: " + modelo.saidaParaArray()[0]
         );
      }
      
      System.out.println("\nPerda: " + modelo.avaliar(entrada, saida));
   }

   /**
    * Salva um arquivo csv com o historico de desempenho do modelo.
    * @param modelo modelo.
    * @param caminho caminho onde será salvo o arquivo.
    */
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
}
