import java.awt.image.BufferedImage;
import java.util.concurrent.TimeUnit;
import ged.Ged;
import geim.Geim;
import rna.avaliacao.perda.*;
import rna.core.Mat;
import rna.estrutura.*;
import rna.inicializadores.*;
import rna.modelos.Sequencial;
import rna.otimizadores.*;
import rna.serializacao.Serializador;

public class MainConv{
   static Ged ged = new Ged();
   static Geim geim = new Geim();

   static final int NUM_DIGITOS = 2;
   static final int NUM_AMOSTRAS = 10;

   public static void main(String[] args){
      ged.limparConsole();
      
      double[][][][] entradas = new double[NUM_AMOSTRAS * NUM_DIGITOS][NUM_DIGITOS][][];
      double[][] saidas = new double[NUM_AMOSTRAS * NUM_DIGITOS][NUM_DIGITOS];
      entradas = carregarDadosMNIST(NUM_AMOSTRAS, NUM_DIGITOS);
      System.out.println("Imagens carregadas. (" + entradas.length + ")");
      saidas = carregarRotulosMNIST(NUM_AMOSTRAS, NUM_DIGITOS);
      System.out.println("Rótulos carregados.");

      Sequencial modelo = criarModelo();
      System.out.println(modelo.info());

      //treinar e marcar tempo
      long t1, t2;
      long horas, minutos, segundos;

      System.out.println("Treinando.");
      t1 = System.nanoTime();
      modelo.treinar(entradas, saidas, 40);
      t2 = System.nanoTime();
      
      long tempoDecorrido = t2 - t1;
      long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempoDecorrido);
      horas = segundosTotais / 3600;
      minutos = (segundosTotais % 3600) / 60;
      segundos = segundosTotais % 60;
      System.out.println("Tempo de treinamento: " + horas + "h " + minutos + "m " + segundos + "s");
      System.out.println("Perda: " + modelo.avaliador.entropiaCruzada(entradas, saidas));
      testes.modelos.TesteModelos.exportarHistoricoPerda(modelo);

      for(int i = 0; i < NUM_DIGITOS; i++){
         testarPorbabilidade(modelo, (i + "_teste_1"));
      }

      // salvarSequencial(modelo, "./modelo-convolucional.txt");
      Main.executarComando("python grafico.py");
   }

   public static Sequencial criarModelo(){
      int[] formEntrada = {28, 28, 1};
      
      Sequencial modelo = new Sequencial(new Camada[]{
         new Convolucional(formEntrada, new int[]{4, 4}, 5, "leakyrelu"),
         new Flatten(),
         new Densa(100, "leakyrelu"),
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
    * 
    * @param caminho
    * @return
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
   public static double[][][][] carregarDadosMNIST(int amostras, int digitos){
      String caminho = "/dados/mnist/treino/";
      double[][][][] entradas = new double[digitos * amostras][1][][];

      int id = 0;
      for(int i = 0; i < digitos; i++){
         for(int j = 0; j < amostras; j++){
            String caminhoCompleto = caminho + i + "/img_" + j + ".jpg";
            double[][] imagem = imagemParaMatriz(caminhoCompleto);
            entradas[id++][0] = imagem;
         }
      }

      return entradas;
   }

   public static double[][] carregarRotulosMNIST(int amostras, int digitos){
      double[][] rotulos = new double[digitos * amostras][digitos];
      for(int numero = 0; numero < digitos; numero++){
         for(int i = 0; i < amostras; i++){
            int indice = numero * amostras + i;
            rotulos[indice][numero] = 1;
         }
      }
  
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

   static boolean compararMatrizes(Mat a, Mat b){
      for(int i = 0; i < a.lin(); i++){
         for(int j = 0; j < a.col(); j++){
            if(a.dado(i, j) != b.dado(i, j)) return false;
         }
      }
      return true;
   }

   static void salvarSequencial(Sequencial modelo, String caminho){
      Serializador s = new Serializador();
      s.salvar(modelo, caminho);
   }
}
