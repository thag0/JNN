package lib.geim;

import java.awt.image.BufferedImage;
import rna.modelos.RedeNeural;

/**
 * <p>
 *    Gerenciador de Imagens.
 * </p>
 * Oferece utilitários para manipulação e processamento de imagens em formato RGB ou escala de cinza.
 * Permite ler imagens, gerar estruturas de dados a partir das imagens, configurar cores, exibir informações 
 * de cores, exportar imagens em formato PNG e ampliar imagens usando uma rede neural treinada.
 */
public class Geim{

   private GerenciadorArquivos ga;
   private GerenciadorDadosImagem gdi;

   /**
    * Objeto responsável por fazer operações com imagens.
    * <p>
    *    Trabalha com imagens no formato {@code Pixel[][];}
    * </p>
    */
   public Geim(){
      ga = new GerenciadorArquivos();
      gdi = new GerenciadorDadosImagem();
   }


   /**
    * Lê uma imagem do caminho fornecido e retorna a imagem como um objeto BufferedImage.
    * @param caminho o caminho da imagem a ser lida. Deve ser um caminho relativo ou absoluto para o arquivo de imagem, deve
    * conter a extensão do arquivo.
    * @return a imagem lida como um objeto BufferedImage.
    * @throws IllegalArgumentException se ocorrer um erro durante a leitura da imagem ou se a imagem não puder ser encontrada.
    */
   public BufferedImage lerImagem(String caminho){
      return ga.lerImagem(caminho);
   }


   /**
    * Gera uma estrutura de dados do tipo {@code Pixel[][]} contendo as informações de cada cor 
    * em cada pixel da imagem.
    * <p>
    *    Todos os valores de cores da imagem serão copiados para a estrutura de dados.
    * </p>
    * @param imagem imagem com suas dimensões e cores.
    * @return estrutura de dados baseada na imagem.
    * @throws IllegalArgumentException se a imagem for nula.
    * @throws IllegalArgumentException se a largura da imagem for menor ou igual a zero.
    * @throws IllegalArgumentException se a altura da imagem for menor ou igual a zero.
    */
   public Pixel[][] gerarEstruturaImagem(BufferedImage imagem){
      return gdi.gerarEstruturaImagem(imagem);
   }


   /**
    * Gera uma estrutura de dados do tipo {@code Pixel[][]}, a estrutura terá seus elementos inicializados
    * como 0.
    * @param largura largura desejada para a estrutura da imagem.
    * @param altura altura desejada para a estrutura da imagem.
    * @return estrutura de dados baseada no tamanho fornecido.
    * @throws IllegalArgumentException se os valores de altura e largura forem menores ou iguais a zero.
    */
   public Pixel[][] gerarEstruturaImagem(int largura, int altura){
      return gdi.gerarEstruturaImagem(largura, altura);
   }


   /**
    * Define a configuração de cor RGB em um pixel específico da estrutura da imagem.
    * @param estruturaImagem estrutura de dados da imagem.
    * @param x coordenada x do pixel.
    * @param y cooredana y do pixel.
    * @param r valor de intensidade da cor vermelha no pixel especificado.
    * @param g valor de intensidade da cor verde no pixel especificado.
    * @param b valor de intensidade da cor azul no pixel especificado.
    * @throws IllegalArgumentException se a estrutura de dados da imagem estiver nula.
    * @throws IllegalArgumentException se o valor de x ou y estiver fora dos índices válidos de acordo 
    * com o tamanho da estrutura da imagem.
    */
   public void configurarCor(Pixel[][] estruturaImagem, int x, int y, int r, int g, int b){
      gdi.configurarCor(estruturaImagem, x, y, r, g, b);
   }

   /**
    * Captura o valor de cor vermelha de cada pixel da imagem.
    * @param imagem imagem com os valore de cores.
    * @return matriz com o valor de cor vermelha de cada pixel.
    */
   public int[][] obterVermelho(BufferedImage imagem){
      return gdi.obterVermelho(imagem);
   }

   /**
    * Captura o valor de cor verde de cada pixel da imagem.
    * @param imagem imagem com os valore de cores.
    * @return matriz com o valor de cor verde de cada pixel.
    */
   public int[][] obterVerde(BufferedImage imagem){
      return gdi.obterVerde(imagem);
   }

   /**
    * Captura o valor de cor azul de cada pixel da imagem.
    * @param imagem imagem com os valore de cores.
    * @return matriz com o valor de cor azul de cada pixel.
    */
   public int[][] obterAzul(BufferedImage imagem){
      return gdi.obterAzul(imagem);
   }

   /**
    * Captura o valor de escala de cinza de cada pixel da imagem.
    * @param imagem imagem com os valore de cores.
    * @return matriz com o valor de escala de cinza de cada pixel.
    */
   public int[][] obterCinza(BufferedImage imagem){
      return gdi.obterCinza(imagem);
   }

   /**
    * Exibe via terminal os valores de intensidade de cor vermelha, verde e azul de cada elemento da estrutura da imagem.
    * @param estruturaImagem estrutura de dados da imagem.
    */
   public void exibirImagemRGB(Pixel[][] estruturaImagem){

      String buffer;
      for(int y = 0; y < estruturaImagem.length; y++){
         for(int x = 0; x < estruturaImagem[y].length; x++){
            buffer = "[r: " + String.valueOf(estruturaImagem[y][x].getR()) + " ";
            buffer += "g: " + String.valueOf(estruturaImagem[y][x].getG()) + " ";
            buffer += "b: " + String.valueOf(estruturaImagem[y][x].getB()) + "] ";
            System.out.print(buffer);
         }
         System.out.println();
      }
   }


   /**
    * Salva a estrutura de imagem em um arquivo de imagem png.
    * @param estruturaImagem estrutura de dados da imagem.
    * @param caminho caminho relativo, deve conter o nome do arquivo, sem extensão
    */
   public void exportarImagemPng(Pixel[][] estruturaImagem, String caminho){
      ga.exportarImagemPng(estruturaImagem, caminho);
   }

   /**
    * Captura os dados das posições dos pixeis e seus valores de cor rgb,
    * de acordo com o seguinte formato:
    * <pre>
    *    linha = [x][y][r][g][b]
    * </pre>
    * Cada linha representará a informação de um pixel individual.
    * @param imagem imagem desejada
    * @return estrutura de dados baseada na imagem.
    */
   public int[][] obterDadosImagem(BufferedImage imagem){
      return gdi.obterDadosImagem(imagem);
   }

   /**
    * Converte a imagem em uma matriz de dados para treino.
    * A matriz terá três colunas, correspondente a posição x normalizada, posição y normalizada e valor da escala de cinza normalizada.
    * <p>
    *    Os valores são normalizados numa escala entre 0 e 1, onde quanto mais próximo de 1 o valor for, mais próximo do tamanho original ele
    *    está.
    * </p>
    * <p> 
    *    Exemplificando que temos uma imagem 10x10 e o valor x  do pixel é igual a 5, o valor normalizado será de 0.5 ou 50% do tamanho 
    *    normalizado na direção horizontal.
    * </p>
    * A organização da matriz seguirá a seguinte estrutura, tendo x, y e escala de cinza normalizados:
    * <p>
    *    {@code [ x pixel ][ y pixel ][ escala de cinza ]}
    * </p>
    * @param imagem imagem original.
    * @return matriz da estrutura de dados da imagem, com os valores normalizados da posição x e y do pixel e escala de cinza.
    * @throws IllegalArgumentException se a imagem for nula.   
    */
   public double[][] imagemParaDadosTreinoEscalaCinza(BufferedImage imagem){
      return gdi.imagemParaDadosTreinoEscalaCinza(imagem);
   }

   /**
    * Converte a imagem em uma matriz de dados para treino contendo seu valor de cor RGB.
    * A matriz terá cinco colunas, correspondente a posição x normalizada, posição y normalizada e valor rgb decomposto na cor vermelha, verde e azul normalizados.
    * <p>
    *    Os valores são normalizados numa escala entre 0 e 1, onde quanto mais próximo de 1 o valor for, mais próximo do tamanho original ele
    *    está.
    * </p>
    * <p>
    *    Para cada cor, será usado o valor de máximo e mínimo como 255.
    * </p>
    * <p> 
    *    Exemplificando que temos uma imagem 10x10 e o valor x  do pixel é igual a 5, o valor normalizado será de 0.5 ou 50% do tamanho 
    *    normalizado na direção horizontal.
    * </p>
    *    A organização da matriz seguirá a seguinte estrutura, tendo x, y e as cores rgb normalizadas:
    * <p>
    *    {@code [ x pixel ][ y pixel ][ vermelho ][ verde ][ azul ]}
    * </p>
    * @param imagem imagem iriginal em escala de cinza.
    * @return matriz da estrutura de dados da imagem, com os valores normalizados da posição x e y, além dos valores de cor rgb do pixel.
    * @throws IllegalArgumentException se a imagem for nula.   
    */
   public double[][] imagemParaDadosTreinoRGB(BufferedImage imagem){
      return gdi.imagemParaDadosTreinoRGB(imagem);
   }

   /**
    * Salva a imagem em escala de cinza a partir de uma rede neural treianda em um arquivo .png no caminho especificado.
    *
    * @param imagem imagem original em escala de cinza que será salva.
    * @param rede rede neural treinada para lidar com a imagem.
    * @param escala escala de ampliação da nova imagem.
    * @param caminho caminho onde o arquivo será salvo, incluindo nome, sem a extensão .png.
    * @throws IllegalArgumentException se a imagem for nula.
    * @throws IllegalArgumentException se o valor de escala for menor ou igual a 0.
    * @throws IllegalArgumentException se a rede tiver um neurônio na camada de saída para tratar a escala de cinza.
    */
   public void exportarImagemEscalaCinza(BufferedImage imagem, RedeNeural rede, float escala, String caminho){
      ga.exportarImagemEscalaCinza(gdi, imagem, rede, escala, caminho);
   }

   /**
    * Salva a imagem com padrão de cor rgb a partir de uma rede neural treianda em um arquivo .png no caminho especificado.
    *
    * @param imagem imagem original em RGB que será salva.
    * @param rede rede neural treinada para lidar com a imagem.
    * @param escala escala de ampliação da nova imagem.
    * @param caminho caminho onde o arquivo será salvo, incluindo nome, sem a extensão .png.
    * @throws IllegalArgumentException se a imagem for nula.
    * @throws IllegalArgumentException se o valor de escala for menor ou igual a 0.
    * @throws IllegalArgumentException se a rede não tiver três neurônios na camada de saída para tratar RGB.
    */
   public void exportarImagemRGB(BufferedImage imagem, RedeNeural rede, float escala, String caminho){
      ga.exportarImagemRGB(gdi, imagem, rede, escala, caminho);
   }
}
