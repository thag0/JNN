package rna.camadas;

import java.util.Random;

import rna.core.Mat;
import rna.inicializadores.Inicializador;

/**
 * Implementação em adamento ainda
 */
public class Dropout extends Camada implements Cloneable{

   private double taxa;
   public Mat entrada;
   public Mat mascara;
   public Mat saida;
   public Mat gradEntrada;
   Random random = new Random();//talvez implementar configuração de seed

   /**
    * @param taxa 
    */
   public Dropout(double taxa){
      if(taxa < 0 || taxa > 1){
         throw new IllegalArgumentException(
            "O valor da taxa de dropout deve estar entre 0 e 1, " + 
            "recebido: " + taxa
         );
      }

      this.taxa = taxa;
   }

   /**
    * @param taxa 
    * @param seed 
    */
   public Dropout(double taxa, long seed){
      if(taxa < 0 || taxa > 1){
         throw new IllegalArgumentException(
            "O valor da taxa de dropout deve estar entre 0 e 1, " + 
            "recebido: " + taxa
         );
      }

      this.taxa = taxa;
      this.random.setSeed(seed);
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
      this.mascara = new Mat(formatoEntrada[0], formatoEntrada[1]);
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
      super.verificarConstrucao();

      if(entrada instanceof Mat == false){
         throw new IllegalArgumentException(
            "Entrada aceita para a camada de Dropout deve ser do tipo Mat, " + 
            "objeto recebido é do tipo \"" + entrada.getClass().getTypeName() + "\"."
         );
      }
      this.entrada.copiar((Mat) entrada);

      if(this.treinando){
         gerarMascara();

         int lin = this.saida.lin(), col = this.saida.col();
         double val = 0;
         for(int i = 0; i < lin; i++){
            for(int j = 0; j < col; j++){
               val = this.mascara.elemento(i, j) * this.entrada.elemento(i, j);
               this.saida.editar(i, j, val);
            }
         }

      }else{
         this.saida.copiar((Mat) entrada);
      }
   }

   private void gerarMascara(){
      for(int i = 0; i < this.mascara.lin(); i++){
         for(int j = 0; j < this.mascara.col(); j++){
            double val = random.nextDouble() > this.taxa ? 1 : 0;
            this.mascara.editar(i, j, val);
         }
      }
   }

   @Override
   public void calcularGradiente(Object gradSeguinte){
      super.verificarConstrucao();

      if(gradSeguinte instanceof Mat == false){
         throw new IllegalArgumentException(
            "Gradiente aceito para a camada de Dropout deve ser do tipo Mat, " + 
            "objeto recebido é do tipo \"" + entrada.getClass().getTypeName() + "\"."
         );
      }

      Mat grads = (Mat) gradSeguinte;
      this.gradEntrada.copiar(grads);
      this.gradEntrada.mult(this.mascara);
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

   /**
    * Retorna a taxa de dropout usada pela camada.
    * @return taxa de dropout da camada.
    */
   public double taxa(){
      return this.taxa;

   }

   @Override
   public Dropout clonar(){
      try{
         Dropout clone = (Dropout) super.clone();
         clone.taxa = this.taxa;
         clone.random = new Random();

         clone.entrada = this.entrada.clone();
         clone.mascara = this.mascara.clone();
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
