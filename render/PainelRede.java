package render;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.ArrayList;

import javax.swing.JPanel;

import rna.estrutura.CamadaDensa;
import rna.estrutura.RedeNeural;

public class PainelRede extends JPanel{
   final int largura;
   final int altura;

   Graphics2D g2;
   
   public RedeNeural rede;

   //coordenadas de origem dos neuronios
   ArrayList<ArrayList<Coordenada>> coordenadas = new ArrayList<>();
   int contadorCoordenadas = 0;
   int contadorCamadas = 0;

   //desenho
   int x = 0;
   int y = 0;
   int larguraDesenho = 28;
   int alturaDesenho = larguraDesenho;
   int padNeuronios = 8;// espaçamento vertical entre os neuronios
   int padCamadas = (int)(larguraDesenho * 2.5);// espaçamento horizontal entre as camadas
   String texto = "";

   //cores
   int r = 150;
   int g = 180;
   int b = 210;
   Color corNeuronioPositivo = new Color(r, g, b);
   Color corNeuronioNegativo = new Color((int)(r * 0.4), (int)(g * 0.4), (int)(b * 0.4));
   Color corBordaNeuronio = Color.BLACK;

   Color corConexaoPositiva = new Color((int)(r * 0.7), (int)(g * 0.7), (int)(b * 0.7));
   Color corConexaoNegativa = new Color((int)(r * 0.25), (int)(g * 0.25), (int)(b * 0.25));

   Color corTexto = new Color(r, g, b);

   //auxiliares
   BasicStroke linhaDesenho = new BasicStroke(1.9f);

   public PainelRede(int largura, int altura){
      this.largura = largura;
      this.altura = altura;
      configInicial(this.largura, this.altura);
   }

   public PainelRede(){
      this.largura = 800;
      this.altura = 600;
      configInicial(this.largura, this.altura);
   }
   
   void configInicial(int largura, int altura){
      setPreferredSize(new Dimension(largura, altura));
      setBackground(new Color(23, 23, 23));
      setFocusable(true);
      setDoubleBuffered(true);
      setEnabled(true);
      setVisible(true);

      // rede base pra nao dar problema
      int arq[] = {1, 1};
      this.rede = new RedeNeural(arq);
      this.rede.compilar();

      // this.larguraDesenho = (int) (this.largura * this.altura * 0.00005);
      // this.alturaDesenho = larguraDesenho;
      // this.padCamadas = (int)(larguraDesenho * 2.5);
   }

   public void desenhar(RedeNeural rede){
      this.rede = rede;
      repaint();
   }

   public void configurarRede(RedeNeural rede){
      this.rede = rede;
   }
   
   @Override
   protected void paintComponent(Graphics g){
      super.paintComponent(g);
      g2 = (Graphics2D) g;
   
      g2.setFont(getFont().deriveFont(14.f));
      contadorCoordenadas = 0;
      contadorCamadas = 0;
      
      //esse alinhamento ainda fica um pouco torto mas é o melhor que deu pra fazer
      int numCamadas = rede.obterQuantidadeCamadas();
      int larguraTotal = (numCamadas * larguraDesenho * 2);
      int xInicial = (largura/2) - (larguraTotal/2);

      x = xInicial;
      for (CamadaDensa camada : rede.obterCamadas()){
         desenharCamada(camada);
         x += padCamadas;
         contadorCamadas++;
      }

      desenhaConexoes(g2);

      g2.dispose();
   }
    
   private void desenharCamada(CamadaDensa camada){
      int alturaJanela = getHeight();
      //tirar altura o padding do ultimo neuronio porque não muda nada e descentraliza o desenho
      int alturaCamada = (camada.quantidadeNeuronios() * (alturaDesenho + padNeuronios)) - padNeuronios;
      int yInicial = (alturaJanela - alturaCamada) / 2;
   
      int y = yInicial;
   
      int neuronios = camada.quantidadeNeuronios();
      for(int i = 0; i < neuronios; i++){
         desenharNeuronio(g2, x, y, camada.saida[0][i]);
         y += alturaDesenho + padNeuronios;
      }
   }
   
   private void desenhaConexoes(Graphics2D g2){
      g2.setStroke(linhaDesenho);

      for(int i = 0; i < coordenadas.size()-1; i++){
         ArrayList<Coordenada> camadaAtual = coordenadas.get(i);
         ArrayList<Coordenada> proximaCamada = coordenadas.get(i+1);

         for(Coordenada c1 : camadaAtual){
            for(Coordenada c2 : proximaCamada){
               if(c1.valor > 0 && c2.valor > 0) g2.setColor(corConexaoPositiva);
               else g2.setColor(corConexaoNegativa);

               //alinhamento das conexões com a borda do neurônio
               int x1 = c1.x + (larguraDesenho);
               int x2 = c2.x + (larguraDesenho * (3/4));
               int y1 = c1.y + (alturaDesenho/2);
               int y2 = c2.y + (alturaDesenho/2);

               g2.drawLine(x1, y1, x2, y2);
            }
         }
      }
   }
  
   private void desenharNeuronio(Graphics2D g2, int x, int y, double valorSaida){
      g2.setColor(corBordaNeuronio);
      g2.fillOval(x, y, larguraDesenho, alturaDesenho);

      int borda = 3;
      
      Coordenada coordenada = new Coordenada(x, y, valorSaida);
      
      if(contadorCamadas < coordenadas.size()){
         coordenadas.get(contadorCamadas).add(coordenada);
         
      }else{
         ArrayList<Coordenada> novaLista = new ArrayList<>();
         novaLista.add(coordenada);
         coordenadas.add(novaLista);
      }
      
      if(valorSaida > 0) g2.setColor(corNeuronioPositivo);
      else g2.setColor(corNeuronioNegativo);
      g2.fillOval(x+borda, y+borda, larguraDesenho-(borda*2), alturaDesenho-(borda*2));  
   }
}