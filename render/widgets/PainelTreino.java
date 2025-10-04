package render.widgets;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jnn.core.tensor.Tensor;
import jnn.core.tensor.Variavel;
import jnn.modelos.Modelo;

public class PainelTreino extends Widget {

   Modelo modelo;

   BufferedImage imagem;
   int epocaAtual = 0;

   //evitar inicializações durante a renderização
   int r, b, g, rgb;
   int x, y;
   
   public PainelTreino(int largura, int altura, double escala) {
      super((int) (escala * largura), (int) (escala * altura));

      imagem = new BufferedImage(this.largura, this.altura, BufferedImage.TYPE_INT_RGB);

      setBackground(new Color(30, 30, 30));
      setFocusable(true);
      setEnabled(true);
      setVisible(true);

      setDoubleBuffered(true);
   }

   public void desenhar(Modelo modelo, int epocasPorFrame) {
      this.modelo = modelo;
      int nSaida = modelo.camadaSaida().tamSaida();

      Tensor in = new Tensor(2);
      
      
      if (nSaida == 1) {//escala de cinza
         Variavel[] saida = modelo.saidaParaArray();//saida 1D
         Variavel s = saida[0];

         for (y = 0; y < this.altura; y++) {
            for (x = 0; x < this.largura; x++) {
               in.set(((double)x / this.largura), 0);
               in.set(((double)y / this.altura), 0);

               modelo.forward(in);

               int cinza = (int)(s.get() * 255);

               r = cinza;
               g = cinza;
               b = cinza;
               rgb = (r << 16) | (g << 8) | b;
               imagem.setRGB(x, y, rgb);
            }
         } 

      } else if (nSaida == 3) {//rgb
         Variavel[] saida = modelo.saidaParaArray();//saida 3D
         Variavel saidaR = saida[0];
         Variavel saidaG = saida[1];
         Variavel saidaB = saida[2];
         
         for (y = 0; y < this.altura; y++) {
            for (x = 0; x < this.largura; x++) {
               in.set(((double)x / this.largura), 0);
               in.set(((double)y / this.altura), 0);

               modelo.forward(in);

               r = (int)(saidaR.get() * 255);
               g = (int)(saidaG.get() * 255);
               b = (int)(saidaB.get() * 255);
               rgb = (r << 16) | (g << 8) | b;
               imagem.setRGB(x, y, rgb);
            }
         }
      }

      epocaAtual = epocasPorFrame;
      repaint();
   }

   public void desenhar(Modelo modelo, int epocasPorFrame, int numThreads) {
      int numSaidas = modelo.camadaSaida().tamSaida();
      
      Modelo[] clones = new Modelo[numThreads];
      try (ExecutorService exec = Executors.newFixedThreadPool(numThreads)) {
         for (int i = 0; i < numThreads; i++) {
            final int id = i;
            clones[id] = modelo.clone();
   
            exec.submit(() -> {
               for (int j = id; j < altura; j += numThreads) {
                  if (numSaidas == 1) calcCinza(clones[id], j);
                  else if (numSaidas == 3) calcRgb(clones[id], j);
               }
            });
         }
      }
      
      epocaAtual = epocasPorFrame;
      repaint();
   }
   
   private void calcCinza(Modelo modelo, int y) {
      Tensor in = new Tensor(2);
      in.set(((double) y / altura), 1);

      int[] pixels = new int[largura];
      int cinza, r, g, b;

      for (int x = 0; x < largura; x++) {
         in.set(((double) x / largura), 0);
         cinza = (int)(modelo.forward(in).get(0) * 255);
            
         r = cinza;
         g = cinza;
         b = cinza;
         pixels[x] = (r << 16) | (g << 8) | b;
      }

      imagem.setRGB(0, y, largura, 1, pixels, 0, largura);
   }

   private void calcRgb(Modelo modelo, int y) {
      Tensor in = new Tensor(2);
      in.set(((double) y / altura), 1);

      int[] pixels = new int[largura];
      double[] saida = new double[3];
      for (int x = 0; x < largura; x++) {
         in.set(((double) x / largura), 0);
         
         modelo.forward(in);
         modelo.copiarDaSaida(saida);

         int r = (int) (saida[0] * 255);
         int g = (int) (saida[1] * 255);
         int b = (int) (saida[2] * 255);
         pixels[x] = (r << 16) | (g << 8) | b;
      }
      
      imagem.setRGB(0, y, largura, 1, pixels, 0, largura);
   }

   @Override
   protected void paintComponent(Graphics g) {
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

