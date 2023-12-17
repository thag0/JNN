package render;

import java.awt.image.BufferedImage;
import java.io.File;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

import rna.modelos.RedeNeural;

public class JanelaTreino extends JFrame{

   public PainelTreino painelTreino;
   private int numThreads = 1;

   public JanelaTreino(int larguraImagem, int alturaImagem, float escala, int numThreads){
      try{
         BufferedImage icone = ImageIO.read(new File("./render/rede-neural.png"));
         setIconImage(icone);
      }catch(Exception e){}

      this.painelTreino = new PainelTreino(larguraImagem, alturaImagem, escala);
      
      setTitle("Treinamento rede");
      add(painelTreino);
      setVisible(true);
      pack();
      setResizable(false);
      setLocationRelativeTo(null);

      if(numThreads < 1){
         throw new IllegalArgumentException(
            "O número de threads deve ser maior que zero."
         );
      }
      this.numThreads = numThreads;
   }


   public void desenharTreino(RedeNeural rede, int epocasPorFrame){
      if(this.numThreads == 1){
         painelTreino.desenhar(rede, epocasPorFrame);
      }else{
         painelTreino.desenharMultithread(rede, epocasPorFrame, this.numThreads);
      }
   }
}
