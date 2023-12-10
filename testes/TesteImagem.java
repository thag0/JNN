package testes;

import java.awt.image.BufferedImage;

import ged.Ged;
import geim.Geim;
import rna.avaliacao.perda.ErroMedioQuadrado;
import rna.estrutura.*;
import rna.inicializadores.Xavier;
import rna.modelos.Sequencial;
import rna.otimizadores.SGD;

public class TesteImagem{
   
   public static void main(String[] args){
      Ged ged = new Ged();
      Geim geim = new Geim();

      ged.limparConsole();

      //importando imagem para treino da rede
      final String caminho = "/dados/mnist/treino/8/img_0.jpg";
      BufferedImage imagem = geim.lerImagem(caminho);
      double[][] dados = geim.imagemParaDadosTreinoEscalaCinza(imagem);
      int nEntrada = 2;// posição x y do pixel
      int nSaida = 1;// valor de escala de cinza/brilho do pixel

      //preparando dados para treinar a rede
      double[][] dadosEntrada = (double[][]) ged.separarDadosEntrada(dados, nEntrada);
      double[][] dadosSaida = (double[][]) ged.separarDadosSaida(dados, nSaida);

      //criando rede neural para lidar com a imagem
      //nesse exemplo queremos que ela tenha overfitting
      Sequencial modelo = new Sequencial(new Camada[]{
         new Densa(nEntrada, 13, "tanh"),
         new Densa(13, "tanh"),
         new Densa(nSaida, "sigmoid"),
      });
      modelo.compilar(new SGD(0.001, 0.99), new ErroMedioQuadrado(), new Xavier());
      modelo.treinar(dadosEntrada, dadosSaida, 2_000);

      //avaliando resultados
      double precisao = 1 - modelo.avaliador.erroMedioAbsoluto(dadosEntrada, dadosSaida);
      double perda = modelo.avaliador.erroMedioQuadrado(dadosEntrada, dadosSaida);
      System.out.println(modelo.info());
      System.out.println("Precisão = " + (precisao * 100));
      System.out.println("Perda = " + perda);
   }
}
