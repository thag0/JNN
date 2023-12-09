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

      Sequencial seq = new Sequencial(new Camada[]{
         new Densa(2, 3, "tanh"),
         new Densa(1, "sigmoid")
      });
      seq.compilar(new SGD(), new ErroMedioQuadrado(), new Xavier());
      seq.treinar(e, s, 10_000);
      System.out.println("Perda Seq: " + seq.avaliador.erroMedioQuadrado(e, s));
      
      RedeNeural rna = new RedeNeural(new int[]{2, 3, 1});
      rna.compilar(new ErroMedioQuadrado(), new SGD(), new Xavier());
      rna.configurarAtivacao("tanh");
      rna.configurarAtivacao(rna.obterCamadaSaida(), "sigmoid");
      rna.treinar(e, s, 10_000);
      System.out.println("Perda Rna: " + rna.avaliador.erroMedioQuadrado(e, s));
      
      System.out.println();
      for(int i = 0; i < 2; i++){
         for(int j = 0; j < 2; j++){
            double[] amostra = {i, j};

            seq.calcularSaida(amostra);
            rna.calcularSaida(amostra);

            double[] prevSeq = seq.saidaParaArray();
            double[] prevRna = rna.saidaParaArray();
            System.out.println(i + " " + j + " - Rna: " + prevRna[0] + "\t  Seq: " + prevSeq[0]);
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