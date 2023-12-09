package testes;

import ged.Dados;
import ged.Ged;
import rna.avaliacao.perda.ErroMedioQuadrado;
import rna.core.OpMatriz;
import rna.estrutura.Camada;
import rna.estrutura.Densa;
import rna.inicializadores.Xavier;
import rna.modelos.*;
import rna.otimizadores.*;
import rna.treinamento.AuxiliarTreino;

public class TesteSequencial{
   static Ged ged = new Ged();
   static OpMatriz opm = new OpMatriz();

   public static void main(String[] args){
      ged.limparConsole();

      double[][] e = {
         {0, 0},
         {0, 1},
         {1, 0},
         {1, 1}
      };

      double[][] s = {
         {0},
         {1},
         {1},
         {0}
      };

      Sequencial modelo = new Sequencial(new Camada[]{
         new Densa(2, 3, "tanh"),
         new Densa(1, "sigmoid")
      });
      modelo.compilar(new SGD(), new ErroMedioQuadrado(), new Xavier());

      System.out.println(modelo.info());

      modelo.treinar(e, s, 1000);

      for(int i = 0; i < 2; i++){
         for(int j = 0; j < 2; j++){
            double[] amostra = {i, j};
            modelo.calcularSaida(amostra);
            double[] previsao = modelo.saidaParaArray();
            System.out.println(i + " " + j + " = " + previsao[0]);
         }
      }
   }

   /**
    * Salva um arquivo csv com o historico de desempenho da rede.
    * @param modelo modelo de rede neural.
    * @param ged gerenciador de dados.
    */
   public static void exportarHistoricoPerda(Sequencial modelo){
      System.out.println("Exportando histórico de perda");
      double[] perdas = modelo.obterHistorico();
      double[][] dadosPerdas = new double[perdas.length][1];

      for(int i = 0; i < dadosPerdas.length; i++){
         dadosPerdas[i][0] = perdas[i];
      }

      Dados dados = new Dados(dadosPerdas);
      ged.exportarCsv(dados, "historico-perda");
   }
}