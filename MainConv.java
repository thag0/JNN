import java.awt.image.BufferedImage;
import java.util.concurrent.TimeUnit;
import lib.ged.Ged;
import lib.geim.Geim;
import rna.avaliacao.perda.*;
import rna.estrutura.*;
import rna.inicializadores.*;
import rna.modelos.Sequencial;
import rna.otimizadores.*;
import rna.serializacao.Serializador;

public class MainConv{
   static Ged ged = new Ged();
   static Geim geim = new Geim();

   static final int NUM_DIGITOS = 10;
   static final int NUM_AMOSTRAS = 20;

   public static void main(String[] args){
      ged.limparConsole();
      
      final var entradas = carregarDadosMNIST("/dados/mnist/treino/", NUM_AMOSTRAS, NUM_DIGITOS);
      final var saidas = criarRotulosMNIST(NUM_AMOSTRAS, NUM_DIGITOS);

      Sequencial modelo = criarModelo();
      System.out.println(modelo.info());

      // treinar e marcar tempo
      long t1, t2;
      long horas, minutos, segundos;

      System.out.println("Treinando.");
      t1 = System.nanoTime();
      //dedicar uma thread pra executar em segundo plano
      rodarTreino(modelo, entradas, saidas, 60);
      t2 = System.nanoTime();
      
      long tempoDecorrido = t2 - t1;
      long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempoDecorrido);
      horas = segundosTotais / 3600;
      minutos = (segundosTotais % 3600) / 60;
      segundos = segundosTotais % 60;
      System.out.println("Tempo de treinamento: " + horas + "h " + minutos + "m " + segundos + "s");
      System.out.println("Perda: " + modelo.avaliador.entropiaCruzada(entradas, saidas));
      testes.modelos.TesteModelos.exportarHistoricoPerda(modelo);

      salvarSequencial(modelo, "./modelo-convolucional.txt");

      for(int i = 0; i < NUM_DIGITOS; i++){
         testarPorbabilidade(modelo, (i + "_teste_1"));
      }

      Main.executarComando("python grafico.py");
   }

   static void rodarTreino(Sequencial modelo, double[][][][] entradas, double[][] saidas, int epochs){
      Thread t = new Thread(() -> modelo.treinar(entradas, saidas, epochs));
      t.setPriority(Thread.MAX_PRIORITY);
      t.start();
      try{
         t.join();
      }catch(Exception e){
         e.printStackTrace();
      }
   }

   public static Sequencial criarModelo(){
      int[] formEntrada = {28, 28, 1};
      
      Sequencial modelo = new Sequencial(new Camada[]{
         new Convolucional(formEntrada, new int[]{4, 4}, 36, "leakyrelu"),
         new MaxPooling(new int[]{2, 2}),
         new Flatten(),
         new Densa(132, "tanh"),
         new Densa(NUM_DIGITOS, "softmax"),
      });

      modelo.compilar(
         new SGD(0.001, 0.99),
         new EntropiaCruzada(),
         new Xavier(),
         new Zeros()
      );
      modelo.configurarHistorico(true);

      return modelo;
   }

   /**
    * Converte uma imagem numa matriz contendo seus valores de brilho entre 0 e 1.
    * @param caminho caminho da imagem.
    * @return matriz contendo os valores de brilho da imagem.
    */
   public static double[][] imagemParaMatriz(String caminho){
      BufferedImage img = geim.lerImagem(caminho);
      double[][] imagem = new double[img.getHeight()][img.getWidth()];

      int[][] cinza = geim.obterCinza(img);

      for(int y = 0; y < imagem.length; y++){
         for(int x = 0; x < imagem[y].length; x++){
            imagem[y][x] = (double)cinza[y][x] / 255;
         }
      }
      return imagem;
   }

   /**
    * Testa as previsões do modelo no formato de probabilidade.
    * @param modelo modelo sequencial de camadas.
    * @param imagemTeste nome da imagem que deve estar no diretório /minst/teste/
    */
   public static void testarPorbabilidade(Sequencial modelo, String imagemTeste){
      System.out.println("\nTestando: " + imagemTeste);
      double[][][] teste1 = new double[1][][];
      teste1[0] = imagemParaMatriz("/dados/mnist/teste/" + imagemTeste + ".jpg");
      modelo.calcularSaida(teste1);
      double[] previsao = modelo.saidaParaArray();
      for(int i = 0; i < previsao.length; i++){
         System.out.println("Prob: " + i + ": " + (int)(previsao[i]*100) + "%");
      }
   }

   /**
    * 
    * @param amostras quantidade de amostras por dígito
    * @param digitos quantidade de dígitos, iniciando do dígito 0.
    * @return
    */
   public static double[][][][] carregarDadosMNIST(String caminho, int amostras, int digitos){
      double[][][][] entradas = new double[digitos * amostras][1][][];

      int id = 0;
      for(int i = 0; i < digitos; i++){
         for(int j = 0; j < amostras; j++){
            String caminhoCompleto = caminho + i + "/img_" + j + ".jpg";
            double[][] imagem = imagemParaMatriz(caminhoCompleto);
            entradas[id++][0] = imagem;
         }
      }

      System.out.println("Imagens carregadas. (" + entradas.length + ")");
      return entradas;
   }

   public static double[][] criarRotulosMNIST(int amostras, int digitos){
      double[][] rotulos = new double[digitos * amostras][digitos];
      for(int numero = 0; numero < digitos; numero++){
         for(int i = 0; i < amostras; i++){
            int indice = numero * amostras + i;
            rotulos[indice][numero] = 1;
         }
      }
      
      System.out.println("Rótulos gerados.");
      return rotulos;
   }

   static void printImagemMNIST(double[][] sample){
      for(int y = 0; y < sample.length; y++){
         for(int x = 0; x < sample[y].length; x++){
            double v = sample[y][x];
            if(v < 0.5) System.out.print("    ");
            else System.out.print((int)(v*100) + " ");
         }
         System.out.println();
      }
   }

   static void salvarSequencial(Sequencial modelo, String caminho){
      System.out.println("Salvando modelo.");
      Serializador s = new Serializador();
      s.salvar(modelo, caminho, "float");
   }
}
