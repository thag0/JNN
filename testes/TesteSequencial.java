package testes;

import ged.Dados;
import ged.Ged;
import rna.avaliacao.perda.EntropiaCruzada;
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
      /*  1  0  */   
         {0, 1},
         {1, 0},
         {1, 0},
         {0, 1}
      };

      int nEntradas = e[0].length;
      int nSaidas = s[0].length;
      int nOcultas = 10;
      long seed = 12345;
      int epocas = 1_000;

      String atv1 = "tanh";
      String atv2 = "softmax";

      Sequencial seq = new Sequencial(new Camada[]{
         new Densa(nEntradas, nOcultas, atv1),
         new Densa(nOcultas, atv1),
         new Densa(nSaidas, atv2)
      });
      seq.configurarSeed(seed);
      seq.compilar(new SGD(), new EntropiaCruzada(), new Xavier());
      seq.treinar(e, s, epocas);
      double perdaSeq = seq.avaliador.entropiaCruzada(e, s);
      
      RedeNeural rna = new RedeNeural(new int[]{nEntradas, nOcultas, nOcultas, nSaidas});
      rna.configurarSeed(seed);
      rna.compilar(new SGD(), new EntropiaCruzada(), new Xavier());
      rna.configurarAtivacao(atv1);
      rna.configurarAtivacao(rna.camadaSaida(), atv2);
      rna.treinar(e, s, epocas);
      double perdaRna = rna.avaliador.entropiaCruzada(e, s);

      System.out.println("Perda Seq: " + perdaSeq);
      System.out.println("Perda Rna: " + perdaRna);
      System.out.println("Diferença RNA - SEQ = " + (perdaRna - perdaSeq));
      
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
      double[] perdas = modelo.historico();
      double[][] dadosPerdas = new double[perdas.length][1];

      for(int i = 0; i < dadosPerdas.length; i++){
         dadosPerdas[i][0] = perdas[i];
      }

      Dados dados = new Dados(dadosPerdas);
      ged.exportarCsv(dados, "historico-perda");
   }
}