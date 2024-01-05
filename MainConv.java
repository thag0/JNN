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

   static final int NUM_DIGITOS_TREINO = 10;
   static final int NUM_AMOSTRAS_TREINO = 30;
   static final int NUM_DIGITOS_TESTE = 10;
   static final int NUM_AMOSTRAS_TESTE = 10;

   static final String caminhoTreino = "/dados/mnist/treino/";
   static final String caminhoTeste = "/dados/mnist/teste/";
   static final String caminhoSaidaModelo = "./dados/modelosMNIST/modelo-convolucional.txt";

   public static void main(String[] args){
      ged.limparConsole();
      
      final var treinoX = carregarDadosMNIST(caminhoTreino, NUM_AMOSTRAS_TREINO, NUM_DIGITOS_TREINO);
      final var treinoY = criarRotulosMNIST(NUM_AMOSTRAS_TREINO, NUM_DIGITOS_TREINO);
      
      final var testeX = carregarDadosMNIST(caminhoTeste, NUM_AMOSTRAS_TESTE, NUM_DIGITOS_TESTE);
      final var testeY = criarRotulosMNIST(NUM_AMOSTRAS_TESTE, NUM_DIGITOS_TESTE);

      Sequencial modelo = criarModelo();
      modelo.configurarHistorico(true);
      System.out.println(modelo.info());

      // treinar e marcar tempo
      long t1, t2;
      long horas, minutos, segundos;

      System.out.println("Treinando.");
      t1 = System.nanoTime();
      rodarTreino(modelo, treinoX, treinoY, 50);
      t2 = System.nanoTime();
      
      long tempoDecorrido = t2 - t1;
      long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempoDecorrido);
      horas = segundosTotais / 3600;
      minutos = (segundosTotais % 3600) / 60;
      segundos = segundosTotais % 60;
      System.out.println("Tempo de treinamento: " + horas + "h " + minutos + "m " + segundos + "s");
      System.out.println("Perda: " + modelo.avaliador.entropiaCruzada(treinoX, treinoY));
      System.out.println("Acurárcia treino: " + modelo.avaliador.acuracia(treinoX, treinoY));
      System.out.println("Acurárcia teste: " + modelo.avaliador.acuracia(testeX, testeY));
      testes.modelos.TesteModelos.exportarHistoricoPerda(modelo);

      salvarModelo(modelo, caminhoSaidaModelo);
      Main.executarComando("python grafico.py");
   }

   /**
    * 
    * @param modelo
    * @param entradas
    * @param saidas
    * @param epochs
    */
   static void rodarTreino(Sequencial modelo, double[][][][] entradas, double[][] saidas, int epochs){
      try{
         Thread t = new Thread(() -> {
            modelo.treinar(entradas, saidas, epochs, true);
         });
         t.start();
         t.join();
      }catch(Exception e){
         e.printStackTrace();
      }
   }

   /*
    * Criação de modelos para testes.
    */
   static Sequencial criarModelo(){
      int[] formEntrada = {28, 28, 1};
      
      Sequencial modelo = new Sequencial(new Camada[]{
         new Convolucional(formEntrada, new int[]{3, 3}, 38, "leakyrelu"),
         new MaxPooling(new int[]{2, 2}),
         new Convolucional(new int[]{3, 3}, 38, "leakyrelu"),
         new MaxPooling(new int[]{2, 2}),
         new Flatten(),
         new Densa(148, "sigmoid"),
         new Densa(NUM_DIGITOS_TREINO, "softmax"),
      });

      modelo.compilar(
         new SGD(0.0001, 0.999),
         new EntropiaCruzada(),
         new He()
      );

      return modelo;
   }

   /**
    * 
    * @param modelo
    * @param caminho
    */
   static void salvarModelo(Sequencial modelo, String caminho){
      System.out.println("Salvando modelo.");
      Serializador s = new Serializador();
      s.salvar(modelo, caminho, "double");
   }

   /**
    * Converte uma imagem numa matriz contendo seus valores de brilho entre 0 e 1.
    * @param caminho caminho da imagem.
    * @return matriz contendo os valores de brilho da imagem.
    */
   static double[][] imagemParaMatriz(String caminho){
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
   static void testarPorbabilidade(Sequencial modelo, String imagemTeste){
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
   static double[][][][] carregarDadosMNIST(String caminho, int amostras, int digitos){
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

   /**
    * 
    * @param amostras
    * @param digitos
    * @return
    */
   static double[][] criarRotulosMNIST(int amostras, int digitos){
      double[][] rotulos = new double[digitos * amostras][digitos];
      for(int numero = 0; numero < digitos; numero++){
         for(int i = 0; i < amostras; i++){
            int indice = numero * amostras + i;
            rotulos[indice][numero] = 1;
         }
      }
      
      System.out.println("Rótulos gerados de 0 a " + (digitos-1) + ".");
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
}
