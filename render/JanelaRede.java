package render;

import java.awt.image.BufferedImage;
import java.io.File;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

import rna.estrutura.RedeNeural;

public class JanelaRede extends JFrame{

   public PainelRede painel;

   public JanelaRede(int largura, int altura){
      this.painel = new PainelRede(largura, altura);
      configInicial();
   }

   public JanelaRede(){
      this.painel = new PainelRede();
      configInicial();
   }

   void configInicial(){
      try{
         BufferedImage icone = ImageIO.read(new File("./render/rede-neural.png"));
         setIconImage(icone);
      }catch(Exception e){
         
      }
      
      setTitle("Rede neural");
      add(painel);
      setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
      setVisible(true);
      pack();
      setResizable(false);
      setLocationRelativeTo(null);
   }


   public void desenhar(RedeNeural rede){
      painel.desenhar(rede);
   }
}
