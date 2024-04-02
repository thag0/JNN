package testes.modelos;

import lib.ged.Dados;
import lib.ged.Ged;
import rna.camadas.Camada;
import rna.camadas.Densa;
import rna.camadas.Entrada;
import rna.core.OpMatriz;
import rna.modelos.*;

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
      long seed = 123456789L;
      int epocas = 10_000;

      String atv1 = "sigmoid";
      String atv2 = "sigmoid";
      String otm = "adagrad";
      String perda = "mse";

      Sequencial seq = new Sequencial(new Camada[]{
         new Entrada(nEntradas),
         new Densa(nOcultas, atv1),
         new Densa(nOcultas, atv1),
         new Densa(nSaidas, atv2)
      });
      seq.setSeed(seed);
      seq.compilar(otm, perda);
      
      RedeNeural rna = new RedeNeural(new int[]{nEntradas, nOcultas, nOcultas, nSaidas});
      rna.setSeed(seed);
      rna.compilar(otm, perda);
      rna.configurarAtivacao(atv1);
      rna.configurarAtivacao(rna.camadaSaida(), atv2);
      
      seq.treinar(entrada, saida, epocas, false);
      rna.treinar(entrada, saida, epocas, false);

      double perdaSeq = seq.avaliador().erroMedioQuadrado(entrada, saida);
      double perdaRna = rna.avaliador().erroMedioQuadrado(entrada, saida);

      System.out.println("Perda Seq: " + perdaSeq);
      System.out.println("Perda Rna: " + perdaRna);
      System.out.println("Diferença RNA - SEQ = " + (perdaRna - perdaSeq));//esperado 0
      
      System.out.println();
      for(int i = 0; i < 2; i++){
         for(int j = 0; j < 2; j++){
            double[] e = {i, j};
            double[] prevSeq = seq.forward(e).paraArray();
            double[] prevRna = rna.forward(e).paraArray();
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