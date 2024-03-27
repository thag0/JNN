package testes.modelos;

import lib.ged.Dados;
import lib.ged.Ged;
import rna.camadas.*;
import rna.core.Tensor4D;
import rna.core.Utils;
import rna.modelos.*;
import rna.otimizadores.SGD;

public class TesteTreino{
   static Ged ged = new Ged();
   static Utils utils = new Utils();

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

      Tensor4D treinoX = new Tensor4D(entrada);

      Sequencial modelo = new Sequencial(new Camada[]{
         new Entrada(treinoX.dim4()),
         new Densa(3, "sigmoid"),
         new Densa(1, "sigmoid")
      });
      
      modelo.compilar(new SGD(0.0001, 0.9995), "mse");
      modelo.treinar(treinoX, saida, 20_000, false);
      verificar(entrada, saida, modelo);

      double perda = modelo.avaliar(entrada, saida);
      System.out.println("\nPerda: " + perda + "\n");
   }

   static void verificar(Object entrada, double[][] saida, Sequencial modelo){
      Object[] arr = utils.transformarParaArray(entrada);

      for(int i = 0; i < arr.length; i++){
         System.out.println(
            "Real: " + saida[i][0] + ",   " +  
            "Prev: " + modelo.calcularSaida(arr[i]).get(0, 0, 0, 0)
         );
      }
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
