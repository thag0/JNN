package testes.modelos;

import lib.ged.Dados;
import lib.ged.Ged;
import rna.avaliacao.perda.ErroMedioQuadrado;
import rna.core.OpMatriz;
import rna.estrutura.Camada;
import rna.estrutura.Densa;
import rna.inicializadores.*;
import rna.modelos.*;
import rna.otimizadores.*;

public class TesteModelos{
   static Ged ged = new Ged();
   static OpMatriz opm = new OpMatriz();

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

      int nEntradas = entrada[0].length;
      int nSaidas = saida[0].length;
      int nOcultas = 3;
      long seed = 0;
      int epocas = 20_000;

      String atv1 = "tanh";
      String atv2 = "sigmoid";

      double ta = 0.01;
      double m = 0.95;

      Sequencial seq = new Sequencial(new Camada[]{
         new Densa(nEntradas, nOcultas, atv1),
         new Densa(nOcultas, atv1),
         new Densa(nSaidas, atv2)
      });
      seq.configurarSeed(seed);
      seq.compilar(new SGD(ta, m), new ErroMedioQuadrado(), new Xavier(), new Xavier());
      
      RedeNeural rna = new RedeNeural(new int[]{nEntradas, nOcultas, nOcultas, nSaidas});
      rna.configurarSeed(seed);
      rna.compilar(new SGD(ta, m), new ErroMedioQuadrado(), new Xavier(), new Xavier());
      rna.configurarAtivacao(atv1);
      rna.configurarAtivacao(rna.camadaSaida(), atv2);
      
      seq.treinar(entrada, saida, epocas);
      rna.treinar(entrada, saida, epocas);

      double perdaSeq = seq.avaliador.erroMedioQuadrado(entrada, saida);
      double perdaRna = rna.avaliador.erroMedioQuadrado(entrada, saida);

      System.out.println("Perda Seq: " + perdaSeq);
      System.out.println("Perda Rna: " + perdaRna);
      System.out.println("Diferença RNA - SEQ = " + (perdaRna - perdaSeq));
      
      System.out.println();
      for(int i = 0; i < 2; i++){
         for(int j = 0; j < 2; j++){
            seq.calcularSaida(new double[]{i, j});
            rna.calcularSaida(new double[]{i, j});

            double[] prevSeq = seq.saidaParaArray();
            double[] prevRna = rna.saidaParaArray();
            System.out.println(i + " " + j + " - Rna: " + prevRna[0] + "      \t    Seq: " + prevSeq[0]);
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
      double[] perdas = modelo.historico();
      double[][] dadosPerdas = new double[perdas.length][1];

      for(int i = 0; i < dadosPerdas.length; i++){
         dadosPerdas[i][0] = perdas[i];
      }

      Dados dados = new Dados(dadosPerdas);
      ged.exportarCsv(dados, "historico-perda");
   }
}