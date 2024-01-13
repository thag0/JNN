package rna.camadas;

import rna.core.Mat;
import rna.inicializadores.Inicializador;

/**
 * Implementação em adamento ainda
 */
public class Dropout extends Camada implements Cloneable{

   private double taxa;
   public Mat entrada;
   public Mat saida;
   public Mat gradEntrada;

   /**
    * @param taxa 
    */
   public Dropout(double taxa){
      if(taxa < 0 | taxa > 1){
         throw new IllegalArgumentException(
            "O valor da taxa de dropout deve estar entre 0 e 1, " + 
            "recebido: " + taxa
         );
      }

      this.taxa = taxa;
   }

   @Override
   public void construir(Object entrada){
      if(entrada instanceof int[] == false){
         throw new IllegalArgumentException(
            "Objeto esperado para entrada da camada Dropout é do tipo int[], " +
            "objeto recebido é do tipo " + entrada.getClass().getTypeName()
         );
      }

      int[] formatoEntrada = (int[]) entrada;
      if(formatoEntrada.length < 2){
         throw new IllegalArgumentException(
            "O formato de entrada para a camada Dropout deve conter pelo menos dois " + 
            "elementos (altura, largura), objeto recebido possui " + formatoEntrada.length + "."
         );
      }
      if(formatoEntrada[1] < 1 || formatoEntrada[1] < 1){
         throw new IllegalArgumentException(
            "Os valores recebidos para o formato de entrada devem ser maiores que zero, " +
            "recebido = (" + formatoEntrada[0] + ", " + formatoEntrada[1] + ")."
         );
      }

      this.entrada = new Mat(formatoEntrada[0], formatoEntrada[1]);
      this.saida = new Mat(formatoEntrada[0], formatoEntrada[1]);
      this.gradEntrada = new Mat(formatoEntrada[0], formatoEntrada[1]);
      this.construida = true;
   }

   @Override
   public void inicializar(Inicializador iniKernel, Inicializador iniBias, double x){}

   @Override
   public void inicializar(Inicializador iniKernel, double x){}

   @Override
   public void calcularSaida(Object entrada){
      if(entrada instanceof Mat == false){
         throw new IllegalArgumentException(
            "Entrada aceita para a camada de Dropout deve ser do tipo Mat, " + 
            "objeto recebido é do tipo \"" + entrada.getClass().getTypeName() + "\"."
         );
      }

      //TODO implementar a lógica de dropout
      this.saida.copiar((Mat) entrada);
   }

   @Override
   public void calcularGradiente(Object gradSeguinte){
      if(gradSeguinte instanceof Mat == false){
         throw new IllegalArgumentException(
            "Gradiente aceito para a camada de Dropout deve ser do tipo Mat, " + 
            "objeto recebido é do tipo \"" + entrada.getClass().getTypeName() + "\"."
         );
      }

      this.gradEntrada.copiar((Mat) gradSeguinte);
   }

   @Override
   public Object saida(){
      super.verificarConstrucao();
      return this.saida;
   }

   @Override
   public int[] formatoEntrada(){
      super.verificarConstrucao();

      return new int[]{
         this.entrada.lin(),
         this.entrada.col()
      };
   }

   @Override
   public int[] formatoSaida(){
      super.verificarConstrucao();

      return new int[]{
         this.saida.lin(),
         this.saida.col()
      };
   }

   @Override
   public int numParametros(){
      return 0;
   }

   @Override
   public Dropout clonar(){
      try{
         Dropout clone = (Dropout) super.clone();

         clone.taxa = this.taxa;
         clone.construida = this.construida;

         clone.entrada = this.entrada.clone();
         clone.saida = this.saida.clone();
         clone.gradEntrada = this.gradEntrada.clone();

         return clone;
      }catch(Exception e){
         throw new RuntimeException(e);
      }
   }

   @Override
   public Mat obterGradEntrada(){
      super.verificarConstrucao();
      return this.gradEntrada;
   }
}
