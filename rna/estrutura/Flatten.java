package rna.estrutura;

import rna.core.Mat;
import rna.inicializadores.Inicializador;

/**
 * 
 */
public class Flatten extends Camada{
   int[] formEntrada;
   int[] formSaida;

   public Mat[] entrada;
   public double[] saida;

   /**
    * Inicializa uma camada Flatten, que irá achatar a entrada recebida
    * no formato de um array unidimensional.
    * <p>
    *    O formato de entrada da camada deve seguir o formato:
    * </p>
    * <pre>
    *    formEntrada = (altura, largura, profundidade)
    * </pre>
    * @param formEntrada formato dos dados de entrada para a camada.
    */
   public Flatten(int[] formEntrada){
      this.formEntrada = formEntrada;

      int tamanho = 1;
      for(int i : formEntrada){
         tamanho *= i;
      }

      this.entrada = new Mat[formEntrada[2]];
      for(int i = 0; i < this.entrada.length; i++){
         this.entrada[i] = new Mat(formEntrada[0], formEntrada[1]);
      }

      this.saida = new double[tamanho];
   }

   @Override
   public void inicializar(Inicializador iniKernel, Inicializador iniBias, double x){
   }

   @Override
   public void inicializar(Inicializador iniKernel, double x){
   }

   @Override
   public void configurarId(int id){
      this.id = id;
   }

   @Override
   public void calcularSaida(Object entrada){
      if(entrada instanceof double[] == false){
         throw new IllegalArgumentException(
            "Os dados de entrada para a camada Flatten devem ser do tipo \"double[]\", " +
            "objeto recebido é do tipo \"" + entrada.getClass().getTypeName() + "\""
         );
      }

      double[] e = (double[]) entrada;
      System.arraycopy(e, 0, this.saida, 0, this.saida.length);
   }

   @Override
   public void calcularGradiente(Object gradSeguinte){
      if(gradSeguinte instanceof double[]){
         calcularGrad((double[]) gradSeguinte);
      
      }else if(gradSeguinte instanceof Mat){
         calcularGrad((Mat) gradSeguinte);
      
      }else{
         throw new IllegalArgumentException(
            "O gradiente seguinte para a camada Flatten deve ser do tipo \"double[]\" ou \"Mat\", " +
            "Objeto recebido é do tipo " + gradSeguinte.getClass().getTypeName()
         );
      }

   }

   private void calcularGrad(double[] grad){
      int id = 0;

      double[][][] entrada = new double[this.entrada.length][this.entrada[0].lin][this.entrada[0].col];
      for(int i = 0; i < formEntrada[0]; i++){
         for(int j = 0; j < formEntrada[1]; j++){
            for(int k = 0; k < formEntrada[2]; k++){
               entrada[i][j][k] = grad[id++];
            }
         }
      }

      for(int i = 0; i < this.entrada.length; i++){
         this.entrada[i].copiar(entrada[i]);
      }
   }
  
   private void calcularGrad(Mat grad){
      double[] g = grad.paraArray();

      int id = 0;
      for(Mat mat : this.entrada){
         for(int i = 0; i < mat.lin; i++){
            for(int j = 0; j < mat.col; j++){
               mat.editar(i, j, g[id++]);
            }
         }
      }
   }

   @Override
   public double[] saidaParaArray(){
      return this.saida;
   }

   @Override
   public int tamanhoSaida(){
      return this.saida.length;
   }

   @Override
   public Object obterGradEntrada(){
      return this.entrada;
   }

}
