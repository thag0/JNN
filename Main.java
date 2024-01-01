import java.awt.image.BufferedImage;
import java.text.DecimalFormat;
import java.util.concurrent.TimeUnit;

import lib.ged.*;
import lib.geim.Geim;
import render.JanelaTreino;
import rna.avaliacao.perda.*;
import rna.estrutura.*;
import rna.inicializadores.*;
import rna.modelos.Modelo;
import rna.modelos.RedeNeural;
import rna.modelos.Sequencial;
import rna.otimizadores.*;

public class Main{
   static final int epocas = 10*1000;
   static final float escalaRender = 8f;
   static Ged ged = new Ged();
   static Geim geim = new Geim();
   static boolean calcularHistorico = true;
   static final String caminhoImagem = "/dados/mnist/treino/8/img_0.jpg";
   // static final String caminhoImagem = "/dados/mnist/treino/7/img_1.jpg";
   // static final String caminhoImagem = "/dados/32x32/circulos.png";

   public static void main(String[] args){      
      ged.limparConsole();

      int tamEntrada = 2;
      int tamSaida = 1;
      BufferedImage imagem = geim.lerImagem(caminhoImagem);
      
      double[][] dados;
      if(tamSaida == 1) dados = geim.imagemParaDadosTreinoEscalaCinza(imagem);
      else if(tamSaida == 3) dados = geim.imagemParaDadosTreinoRGB(imagem);
      else return;

      double[][] in  = (double[][]) ged.separarDadosEntrada(dados, tamEntrada);
      double[][] out = (double[][]) ged.separarDadosSaida(dados, tamSaida);

      Modelo modelo = criarSequencial(tamEntrada, tamSaida);
      System.out.println(modelo.info());

      //treinar e marcar tempo
      long t1, t2;
      long horas, minutos, segundos;

      System.out.println("Treinando.");
      t1 = System.nanoTime();
      treinoEmPainel(modelo, imagem.getWidth(), imagem.getHeight(), in, out);
      t2 = System.nanoTime();

      long tempoDecorrido = t2 - t1;
      long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempoDecorrido);
      horas = segundosTotais / 3600;
      minutos = (segundosTotais % 3600) / 60;
      segundos = segundosTotais % 60;

      double precisao = (1 - modelo.avaliador.erroMedioQuadrado(in, out))*100;
      double perda = modelo.avaliador.erroMedioQuadrado(in, out);
      System.out.println("Precisão = " + formatarDecimal(precisao, 2) + "%");
      System.out.println("Perda = " + perda);
      System.out.println("Tempo de treinamento: " + horas + "h " + minutos + "m " + segundos + "s");

      if(calcularHistorico){
         exportarHistoricoPerda(modelo);
         executarComando("python grafico.py");
      }
   }

   static Modelo criarRna(int entradas, int saidas){
      Otimizador otm = new SGD(0.0001, 0.9995);
      Perda perda = new ErroMedioQuadrado();
      Inicializador ini = new Xavier();

      int[] arq = {entradas, 13, 13, saidas};
      RedeNeural modelo = new RedeNeural(arq);
      modelo.compilar(otm, perda, ini);
      modelo.configurarAtivacao("tanh");
      modelo.configurarAtivacao(modelo.camadaSaida(), "sigmoid");
      modelo.configurarHistorico(calcularHistorico);
      
      return modelo;
   }

   static Modelo criarSequencial(int entradas, int saidas){
      Otimizador otm = new SGD(0.01, 0.9);
      Perda perda = new ErroMedioQuadrado();
      Inicializador ini = new LeCun();
      
      Sequencial modelo = new Sequencial();
      modelo.add(new Densa(entradas, 13, "tanh"));
      modelo.add(new Densa(13, saidas, "sigmoid"));
      // modelo.configurarSeed(1234);
      modelo.compilar(otm, perda, ini);
      modelo.configurarHistorico(calcularHistorico);

      return modelo;
   }

   /**
    * Treina e exibe o resultado da Rede Neural no painel.
    * @param modelo modelo de rede neural usado no treino.
    * @param altura altura da janela renderizada.
    * @param largura largura da janela renderizada.
    * @param entradas dados de entrada para o treino.
    * @param saidas dados de saída relativos a entrada.
    */
   static void treinoEmPainel(Modelo modelo, int altura, int largura, double[][] entradas, double[][] saidas){
      final int fps = 6000;
      int epocasPorFrame = 30;

      //acelerar o processo de desenho
      //bom em situações de janelas muito grandes
      int n = Runtime.getRuntime().availableProcessors();
      int numThreads = (n > 1) ? (int)(n * 0.5) : 1;

      JanelaTreino jt = new JanelaTreino(largura, altura, escalaRender, numThreads);
      jt.desenharTreino(modelo, 0);
      
      //trabalhar com o tempo de renderização baseado no fps
      double intervaloDesenho = 1000000000/fps;
      double proximoTempoDesenho = System.nanoTime() + intervaloDesenho;
      double tempoRestante;
      
      int i = 0;
      while(i < epocas && jt.isVisible()){
         modelo.treinar(entradas, saidas, epocasPorFrame);
         jt.desenharTreino(modelo, i);
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
   static void exportarHistoricoPerda(Modelo rede){
      System.out.println("Exportando histórico de perda");
      double[] perdas = rede.historico();
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
    * @param casas quantidade de casas após o ponto flutuante.
    * @return
    */
   static String formatarDecimal(double valor, int casas){
      String valorFormatado = "";

      String formato = "#.";
      for(int i = 0; i < casas; i++) formato += "#";

      DecimalFormat df = new DecimalFormat(formato);
      valorFormatado = df.format(valor);

      return valorFormatado;
   }

   /**
    * teste
    * @param comando
    */
   public static void executarComando(String comando){
      try{
         new ProcessBuilder("cmd", "/c", comando).inheritIO().start().waitFor();
      }catch(Exception e){

      }
   }
}
