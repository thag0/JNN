package render;

import java.awt.image.BufferedImage;
import java.io.File;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

import rna.estrutura.RedeNeural;

public class JanelaTreino extends JFrame{

   public PainelTreino painelTreino;

   public JanelaTreino(int larguraImagem, int alturaImagem, float escala){
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
   }


   public void desenharTreino(RedeNeural rede, int epocasPorFrame){
      painelTreino.desenhar(rede, epocasPorFrame);
   }


   /**
    * Acelera o processo de desenho do painel usando o número de threads fornecidas.
    * @param rede
    * @param epocasPorFrame
    * @param numThreads
    */
   public void desenharTreino(RedeNeural rede, int epocasPorFrame, int numThreads){
      if(numThreads <= 0) throw new IllegalArgumentException("O valor do número de threads deve ser maior que 1.");
      
      painelTreino.desenharMultithread(rede, epocasPorFrame, numThreads);
   }
}
