import java.awt.image.BufferedImage;
import java.text.DecimalFormat;
import java.util.concurrent.TimeUnit;

import ged.*;
import geim.Geim;
import render.JanelaTreino;
import rna.avaliacao.perda.*;
import rna.inicializadores.*;
import rna.modelos.RedeNeural;
import rna.otimizadores.*;

public class Main{
   static final int epocas = 5*1000;
   static final float escalaRender = 7f;
   static Ged ged = new Ged();
   static Geim geim = new Geim();

   public static void main(String[] args){
      
      ged.limparConsole();
      BufferedImage imagem = geim.lerImagem("/dados/mnist/treino/7/img_1.jpg");
      double[][] dados = geim.imagemParaDadosTreinoEscalaCinza(imagem);

      int tamEntrada = 2;
      int tamSaida = 1;
      double[][] in  = (double[][]) ged.separarDadosEntrada(dados, tamEntrada);
      double[][] out = (double[][]) ged.separarDadosSaida(dados, tamSaida);

      RedeNeural rede = criarRede(tamEntrada, tamSaida);
      System.out.println(rede.info());

      //treinar e marcar tempo
      long t1, t2;
      long horas, minutos, segundos;

      t1 = System.nanoTime();
      System.out.println("Treinando.");
      treinoEmPainel(rede, imagem, in, out);
      t2 = System.nanoTime();

      long tempoDecorrido = t2 - t1;
      long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempoDecorrido);
      horas = segundosTotais / 3600;
      minutos = (segundosTotais % 3600) / 60;
      segundos = segundosTotais % 60;

      double precisao = (1 - rede.avaliador.erroMedioAbsoluto(in, out))*100;
      double perda = rede.avaliador.erroMedioQuadrado(in, out);
      System.out.println("Precisão = " + formatarDecimal(precisao, 2) + "%");
      System.out.println("Perda = " + perda);
      System.out.println("Tempo de treinamento: " + horas + "h " + minutos + "m " + segundos + "s");
      // exportarHistoricoPerda(rede, ged);
   }

   public static RedeNeural criarRede(int entradas, int saidas){
      int[] arq = {entradas, 13, 13, saidas};//28x28
      RedeNeural rede = new RedeNeural(arq);

      Perda perda = new ErroMedioQuadrado();
      Otimizador otm = new SGD(0.001, 0.95);
      // Otimizador otm = new AdaGrad(0.9999999);
      Inicializador ini = new Xavier();

      // rede.configurarHistoricoPerda(true);
      rede.compilar(perda, otm, ini, ini);
      rede.configurarAtivacao("tanh");
      rede.configurarAtivacao(rede.obterCamadaSaida(), "sigmoid");

      return rede;
   }

   /**
    * Treina e exibe o resultado da Rede Neural no painel.
    * @param rede modelo de rede neural usado no treino.
    * @param imagem imagem que o modelo está aprendendo.
    * @param dadosEntrada dados de entrada para o treino.
    * @param dadosSaida dados de saída relativos a entrada.
    */
   public static void treinoEmPainel(RedeNeural rede, BufferedImage imagem, double[][] dadosEntrada, double[][] dadosSaida){
      final int fps = 600;
      int epocasPorFrame = 30;

      //acelerar o processo de desenho
      //bom em situações de janelas muito grandes
      int numThreads = (int)(Runtime.getRuntime().availableProcessors() * 0.5);

      JanelaTreino jt = new JanelaTreino(imagem.getWidth(), imagem.getHeight(), escalaRender);
      jt.desenharTreino(rede, 0, numThreads);
      
      //trabalhar com o tempo de renderização baseado no fps
      double intervaloDesenho = 1000000000/fps;
      double proximoTempoDesenho = System.nanoTime() + intervaloDesenho;
      double tempoRestante;
      
      int i = 0;
      while(i < epocas && jt.isVisible()){
         rede.treinar(dadosEntrada, dadosSaida, epocasPorFrame);
         jt.desenharTreino(rede, i, numThreads);
         i += epocasPorFrame;

         try{
            tempoRestante = proximoTempoDesenho - System.nanoTime();
            tempoRestante /= 1000000;
            if(tempoRestante < 0) tempoRestante = 0;

            Thread.sleep((long)tempoRestante);
            proximoTempoDesenho += intervaloDesenho;

         }catch(Exception e){ }
      }

      jt.dispose();
   }

   /**
    * Salva um arquivo csv com o historico de desempenho da rede.
    * @param rede rede neural.
    * @param ged gerenciador de dados.
    */
   public static void exportarHistoricoPerda(RedeNeural rede, Ged ged){
      System.out.println("Exportando histórico de perda");
      double[] perdas = rede.obterHistorico();
      double[][] dadosPerdas = new double[perdas.length][1];

      for(int i = 0; i < dadosPerdas.length; i++){
         dadosPerdas[i][0] = perdas[i];
      }

      Dados dados = new Dados(dadosPerdas);
      ged.exportarCsv(dados, "historico-perda");
   }

   /**
    * Formata o valor recebido para a quantidade de casas após o ponto
    * flutuante.
    * @param valor valor alvo.
    * @param casas quantidade de casas após o ponto fluntuante.
    * @return
    */
   public static String formatarDecimal(double valor, int casas){
      String valorFormatado = "";

      String formato = "#.";
      for(int i = 0; i < casas; i++) formato += "#";

      DecimalFormat df = new DecimalFormat(formato);
      valorFormatado = df.format(valor);

      return valorFormatado;
   }
}
