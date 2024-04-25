package lib.geim;

/**
 * Classe responsável por representar o valor de cor de um pixel de 
 * acordo com o padrão de cor rgb.
 */
public class Pixel {

   /**
    * intensidade da cor vermelha.
    */
   private int r;
    
   /**
    * intensidade da cor verde.
    */
   private int g;

   /**
    * intensidade da cor azul.
    */
   private int b;

   /**
    * Cria uma intância de um pixel.
    * <p>
    *    Os valores de intensidade de cada cor serão inicializados de acordo
    *    com os valores fornecidos.
    * </p>
    * @param r intensidade da cor vermelha.
    * @param g intensidade da cor verde.
    * @param b intensidade da cor azul.
    */
   public Pixel(int r, int g, int b) {
      setR(r);
      setG(g);
      setB(b);
   }

   /**
    * Cria uma intância de um pixel.
    * <p>
    *    Os valores de intensidade de cada cor serão inicializados como 0.
    * </p>
    */
   public Pixel() {
      setR(0);
      setG(0);
      setB(0);
   }

   /**
    * Configura o valor de cor vermelha do pixel.
    * <p><strong>
    *    Valores acima de 255 serão automaticamente configurados como 255.
    * </strong></p>
    * <p><strong>
    *    Valores abaixo de 0 serão automaticamente configurados como 0.
    * </strong></p>
    * @param r novo valor de cor vermelha.
    */
   public void setR(int r) {
      if(r > 255) this.r = 255;
      else if(r < 0) this.r = 0;
      else this.r = r;
   }

   /**
    * Configura o valor de cor verde do pixel.
    * <p><strong>
    *    Valores acima de 255 serão automaticamente configurados como 255.
    * </strong></p>
    * <p><strong>
    *    Valores abaixo de 0 serão automaticamente configurados como 0.
    * </strong></p>
    * @param g novo valor de cor verde.
    */
   public void setG(int g) {
      if(g > 255) this.g = 255;
      else if(g < 0) this.g = 0;
      else this.g = g;
   }

   /**
    * Configura o valor de cor azul do pixel.
    * <p><strong>
    *    Valores acima de 255 serão automaticamente configurados como 255.
    * </strong></p>
    * <p><strong>
    *    Valores abaixo de 0 serão automaticamente configurados como 0.
    * </strong></p>
    * @param b novo valor de cor azul.
    */
   public void setB(int b) {
      if(b > 255) this.b = 255;
      else if(b < 0) this.b = 0;
      else this.b = b;
   }

   /**
    * Configura os valores de cor do pixel.
    * @param r intensidade da cor vermelha.
    * @param g intensidade da cor verde.
    * @param b intensidade da cor azul.
    */
   public void setRGB(int r, int g, int b) {
      setR(r);
      setG(g);
      setB(b);
   }

   /**
    * Retorna o valor de intensidade da cor vermelha do pixel.
    * @return valor de intensidade da cor vermelha do pixel.
    */
   public int getR() {
      return this.r;
   }

   /**
    * Retorna o valor de intensidade da cor verde do pixel.
    * @return valor de intensidade da cor verde do pixel.
    */
   public int getG() {
      return this.g;
   }

   /**
    * Retorna o valor de intensidade da cor azul do pixel.
    * @return valor de intensidade da cor azul do pixel.
    */
   public int getB() {
      return this.b;
   }

}
