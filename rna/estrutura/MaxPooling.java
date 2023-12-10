package rna.estrutura;

import rna.core.Mat;
import rna.inicializadores.Inicializador;

public class MaxPooling extends Camada{

   private int[] formEntrada;
   private int[] formSaida;

   private Mat[] entrada;
   private Mat[] saida;
   private Mat[] gradEntrada;

   private int[] formFiltro;
   private int stride;

   public MaxPooling(){

   }

   public MaxPooling(int[] formFiltro, int stride){
      this.stride = stride;
      this.formFiltro = (int[]) formFiltro;
   }

   public MaxPooling(int[] formEntrada, int[] formFiltro, int stride){
      this.stride = stride;
      this.formFiltro = (int[]) formFiltro;
      this.construir(formEntrada);
   }

   /**
    * Instancia uma camada MaxPooling.
    * <p>
    *    O formato de entrada da camada deve seguir o padrão:
    * </p>
    * <pre>
    *    formEntrada = (altura, largura, profundidade)
    * </pre>
    * @param entrada formato dos dados de entrada para a camada.
    */
   @Override
   public void construir(Object entrada){
      if(entrada == null){
         throw new IllegalArgumentException(
            "Formato de entrada fornecida para camada MaxPooling é nulo."
         );
      }
      if(entrada instanceof int[] == false){
         throw new IllegalArgumentException(
            "Objeto esperado para entrada da camada MaxPooling é do tipo int[], " +
            "objeto recebido é do tipo " + entrada.getClass().getTypeName()
         );
      }
      int[] e = (int[]) entrada;
      this.formEntrada = new int[]{e[0], e[1], e[2]};

      this.formSaida = new int[3];
      formSaida[0] = (formEntrada[0] - formFiltro[0]) / stride + 1;
      formSaida[1] = (formEntrada[1] - formFiltro[1]) / stride + 1;
      formSaida[2] = formEntrada[2];
      
      this.entrada = new Mat[formEntrada[2]];
      this.saida = new Mat[formSaida[2]];

      for(int i = 0; i < formEntrada[2]; i++){
         this.entrada[i] = new Mat(this.formEntrada[0], this.formEntrada[1]);
         this.saida[i] = new Mat(this.formSaida[0], this.formSaida[1]);
      }
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
      if(entrada instanceof Mat[]){
         Mat[] e = (Mat[]) entrada;

         for(int i = 0; i < this.entrada.length; i++){
            this.entrada[i].copiar(e[i]);
         }

         for(int i = 0; i < this.entrada.length; i++){
            aplicarMaxPooling(this.entrada[i], this.saida[i]);
         }
      }
   }

   private void aplicarMaxPooling(Mat entrada, Mat saida){
      for(int i = 0; i < saida.lin; i++){
         for(int j = 0; j < saida.col; j++){
            int linInicio = i * this.stride;
            int colIincio = j * this.stride;
            int linFim = Math.min(linInicio + this.formFiltro[0], entrada.lin);
            int colFim = Math.min(colIincio + this.formFiltro[1], entrada.col);
            double maxValor = Double.MIN_VALUE;
            for(int row = linInicio; row < linFim; row++){
               for(int col = colIincio; col < colFim; col++){
                  double valor = entrada.dado(row, col);
                  if (valor > maxValor){
                     maxValor = valor;
                  }
               }
            }
            
            saida.editar(i, j, maxValor);
         }
      }
   }

   @Override
   public void calcularGradiente(Object gradSeguinte) {
       if (gradSeguinte instanceof Mat[]) {
           Mat[] gradCamadaSeguinte = (Mat[]) gradSeguinte;
   
           gradEntrada = new Mat[formEntrada[2]];
   
           for (int i = 0; i < gradEntrada.length; i++) {
               gradEntrada[i] = new Mat(formEntrada[0], formEntrada[1]);
           }
   
           for (int i = 0; i < gradEntrada.length; i++) {
               calcularGradienteMaxPooling(entrada[i], gradCamadaSeguinte[i], gradEntrada[i]);
           }
       }
   }
   
   private void calcularGradienteMaxPooling(Mat entrada, Mat gradCamadaSeguinte, Mat gradEntrada) {
       for (int i = 0; i < gradCamadaSeguinte.lin; i++) {
           for (int j = 0; j < gradCamadaSeguinte.col; j++) {
               int linInicio = i * this.stride;
               int colIincio = j * this.stride;
               int linFim = Math.min(linInicio + this.formFiltro[0], entrada.lin);
               int colFim = Math.min(colIincio + this.formFiltro[1], entrada.col);
   
               for (int row = linInicio; row < linFim; row++) {
                   for (int col = colIincio; col < colFim; col++) {
                       double valorEntrada = entrada.dado(row, col);
                       double valorGradSeguinte = gradCamadaSeguinte.dado(i, j);
   
                       // Se o valor na entrada foi o máximo, propague o gradiente
                       if (valorEntrada == valorGradSeguinte) {
                           gradEntrada.editar(row, col, valorGradSeguinte);
                       }
                   }
               }
           }
       }
   }
   
   @Override
   public int[] formatoSaida(){
      return formSaida;
   }

   @Override
   public int tamanhoSaida(){
      return this.saida.length * this.saida[0].lin * this.saida[0].col;
   }

   @Override
   public double[] saidaParaArray(){
      int id = 0;
      double[] saida = new double[this.tamanhoSaida()];

      for(int i = 0; i < this.saida.length; i++){
         double[] s = this.saida[i].paraArray();
         for(int j = 0; j < s.length; j++){
            saida[id++] = s[j];
         }
      }

      return saida;
   }

   @Override
   public Mat[] obterGradEntrada(){
      return this.gradEntrada;
   }
}
