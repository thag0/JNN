package render;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import javax.swing.JPanel;

import rna.modelos.Modelo;

public class PainelTreino extends JPanel{
   final int largura;
   final int altura;
   Modelo modelo;
   double[] entradaRede;

   BufferedImage imagem;
   int epocaAtual = 0;

   //evitar inicializações durante a renderização
   int r, b, g, rgb;
   int x, y;
   
   public PainelTreino(int larguraImagem, int alturaImagem, float escala){
      this.largura = (int) (larguraImagem*escala);
      this.altura = (int)  (alturaImagem*escala);

      imagem = new BufferedImage(this.largura, this.altura, BufferedImage.TYPE_INT_RGB);

      setPreferredSize(new Dimension(this.largura, this.altura));
      setBackground(new Color(30, 30, 30));
      setFocusable(true);
      setDoubleBuffered(true);
      setEnabled(true);
      setVisible(true);
   }

   public void desenhar(Modelo modelo, int epocasPorFrame){
      this.modelo = modelo;
      
      int nEntrada = 2;
      int nSaida = modelo.camadaSaida().tamanhoSaida();
      entradaRede = new double[nEntrada];

      if(nSaida == 1){//escala de cinza
         for(y = 0; y < this.altura; y++){
            for(x = 0; x < this.largura; x++){
               entradaRede[0] = (double)x / this.largura;
               entradaRede[1] = (double)y / this.altura;
               modelo.calcularSaida(entradaRede);

               double[] saida = modelo.saidaParaArray();
               int cinza = (int)(saida[0] * 255);

               r = cinza;
               g = cinza;
               b = cinza;
               rgb = (r << 16) | (g << 8) | b;
               imagem.setRGB(x, y, rgb);
            }
         } 

      }else if(nSaida == 3){//rgb
         for(y = 0; y < this.altura; y++){
            for(x = 0; x < this.largura; x++){
               entradaRede[0] = (double)x / this.largura;
               entradaRede[1] = (double)y / this.altura;
               modelo.calcularSaida(entradaRede);

               double[] saida = modelo.saidaParaArray();
               r = (int)(saida[0] * 255);
               g = (int)(saida[1] * 255);
               b = (int)(saida[2] * 255);
               rgb = (r << 16) | (g << 8) | b;
               imagem.setRGB(x, y, rgb);
            }
         }
      }

      epocaAtual = epocasPorFrame;
      repaint();
   }

   public void desenharMultithread(Modelo modelo, int epocasPorFrame, int numThreads){
      this.modelo = modelo;
      int nSaida = modelo.camadaSaida().tamanhoSaida();

      //organizar
      Thread[] threads = new Thread[numThreads];
      Modelo[] clones = new Modelo[numThreads];
      for(int i = 0; i < clones.length; i++){
         clones[i] = modelo.clonar();
      }

      int alturaPorThread = this.altura / numThreads;
      int restoAltura = this.altura % numThreads;

      for(int i = 0; i < numThreads; i++){
         final int id = i;
         int inicioY = i * alturaPorThread;
         int fimY = inicioY + alturaPorThread + ((i == numThreads-1) ? restoAltura : 0);

         threads[i] = new Thread(()->{
            if(nSaida == 1){
               calcularParteImagemEscalaCinza(clones[id], inicioY, fimY);
            
            }else if(nSaida == 3){
               calcularParteImagemRGB(clones[id], inicioY, fimY);
            }
         });

         threads[i].start();
      }

      try{
         for(Thread thread : threads){
            thread.join(0);
         }
      }catch(Exception e){
         throw new RuntimeException(e);
      }

      epocaAtual = epocasPorFrame;
      repaint();
   }

   private void calcularParteImagemEscalaCinza(Modelo modelo, int inicioY, int fimY){
      double[] entrada = new double[2];
      double[] saida = new double[modelo.camadaSaida().tamanhoSaida()];

      int r, g, b, rgb, cinza;
      int x, y;

      for(y = inicioY; y < fimY; y++){
         for(x = 0; x < this.largura; x++){
            entrada[0] = (double) x / this.largura;
            entrada[1] = (double) y / this.altura;

            modelo.calcularSaida(entrada);
            
            saida = modelo.saidaParaArray();
            cinza = (int)(saida[0] * 255);
            r = cinza;
            g = cinza;
            b = cinza;
            rgb = (r << 16) | (g << 8) | b;
            imagem.setRGB(x, y, rgb);
         }
      }
   }

   private void calcularParteImagemRGB(Modelo modelo, int inicioY, int fimY){
      double[] entrada = new double[2];
      double[] saida = new double[modelo.camadaSaida().tamanhoSaida()];
      int r, g, b, rgb;

      for(int y = inicioY; y < fimY; y++){
         for (int x = 0; x < this.largura; x++){
            entrada[0] = (double) x / this.largura;
            entrada[1] = (double) y / this.altura;
            
            modelo.calcularSaida(entrada);

            saida = modelo.saidaParaArray();
            r = (int) (saida[0] * 255);
            g = (int) (saida[1] * 255);
            b = (int) (saida[2] * 255);
            rgb = (r << 16) | (g << 8) | b;
            imagem.setRGB(x, y, rgb);
         }
      }
   }

   @Override
   protected void paintComponent(Graphics g){
      super.paintComponent(g);
      Graphics2D g2 = (Graphics2D) g;

      g2.drawImage(imagem, 0, 0, null);

      g2.setFont(getFont().deriveFont(13f));
      
      //efeito de sombra
      g2.setColor(Color.BLACK);
      g2.drawString(("Época: " + epocaAtual), 6, 16);
      g2.drawString(("Época: " + epocaAtual), 4, 16);

      g2.drawString(("Época: " + epocaAtual), 6, 14);
      g2.drawString(("Época: " + epocaAtual), 4, 14);

      g2.setColor(Color.WHITE);
      g2.drawString(("Época: " + epocaAtual), 5, 15);

      g2.dispose();
   }
}

