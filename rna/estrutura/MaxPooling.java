package rna.estrutura;

import rna.core.Mat;
import rna.core.Utils;
import rna.inicializadores.Inicializador;

public class MaxPooling extends Camada{

   /**
    * Utilitario.
    */
   Utils utils = new Utils();

   private int[] formEntrada;
   private int[] formSaida;

   private Mat[] entrada;
   private Mat[] saida;
   private Mat[] gradEntrada;

   private int[] formFiltro;
   private int strideAltura;
   private int strideLargura;

   public MaxPooling(){

   }

   public MaxPooling(int[] formFiltro){
      if(formFiltro[0] != formFiltro[1]){
         throw new IllegalArgumentException(
            "As dimensões do filtro devem ser igual para usar um stride padrão."
         );
      }
      this.strideAltura = formFiltro[0];
      this.strideLargura = formFiltro[0];
      this.formFiltro = (int[]) formFiltro;
   }

   public MaxPooling(int[] formFiltro, int[] stride){
      this.strideAltura = stride[0];
      this.strideLargura = stride[1];
      this.formFiltro = (int[]) formFiltro;
   }

   public MaxPooling(int[] formEntrada, int[] formFiltro, int[] stride){
      this.strideAltura = stride[0];
      this.strideLargura = stride[1];
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
      formSaida[0] = (formEntrada[0] - formFiltro[0]) / this.strideAltura + 1;
      formSaida[1] = (formEntrada[1] - formFiltro[1]) / this.strideLargura + 1;
      formSaida[2] = formEntrada[2];
      
      this.entrada = new Mat[formEntrada[2]];
      this.gradEntrada = new Mat[formSaida[2]];
      this.saida = new Mat[formSaida[2]];

      for(int i = 0; i < formEntrada[2]; i++){
         this.entrada[i] = new Mat(this.formEntrada[0], this.formEntrada[1]);
         this.saida[i] = new Mat(this.formSaida[0], this.formSaida[1]);
         this.gradEntrada[i] = new Mat(this.formEntrada[0], this.formEntrada[1]);
      }
   }

   @Override
   public void inicializar(Inicializador iniKernel, Inicializador iniBias, double x){}

   @Override
   public void inicializar(Inicializador iniKernel, double x){}

   @Override
   public void configurarId(int id){
      this.id = id;
   }

   @Override
   public void calcularSaida(Object entrada){
      if(entrada instanceof Mat[]){
         Mat[] e = (Mat[]) entrada;
         utils.copiar(e, this.entrada);
         for(int i = 0; i < this.entrada.length; i++){
            aplicarMaxPooling(this.entrada[i], this.saida[i]);
         }
         
      }else if(entrada instanceof double[]){
         utils.copiar((double[]) entrada, this.entrada);
         for(int i = 0; i < this.entrada.length; i++){
            aplicarMaxPooling(this.entrada[i], this.saida[i]);
         }

      }else{
         throw new IllegalArgumentException(
            "Tipo de entrada \"" + entrada.getClass().getTypeName() + "\" não suportada."
         );
      }
   }

   private void aplicarMaxPooling(Mat entrada, Mat saida){
      for(int i = 0; i < saida.lin(); i++){
         for(int j = 0; j < saida.col(); j++){
            int linInicio = i * this.strideAltura;
            int colIincio = j * this.strideLargura;
            int linFim = Math.min(linInicio + this.formFiltro[0], entrada.lin());
            int colFim = Math.min(colIincio + this.formFiltro[1], entrada.col());
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
   public void calcularGradiente(Object gradSeguinte){
      if(gradSeguinte instanceof Mat[]){
         Mat[] gradCamadaSeguinte = (Mat[]) gradSeguinte;   
         for(int i = 0; i < gradEntrada.length; i++){
            calcularGradienteMaxPooling(this.entrada[i], gradCamadaSeguinte[i], this.gradEntrada[i]);
         }
      
      }else{
         throw new IllegalArgumentException(
            "Formato de gradiente \" "+ gradSeguinte.getClass().getTypeName() +" \" não " +
            "suportado para camada de MaxPooling."
         );
      }
   }
   
   private void calcularGradienteMaxPooling(Mat entrada, Mat gradSeguinte, Mat gradEntrada){
      for(int i = 0; i < gradSeguinte.lin(); i++){
          for(int j = 0; j < gradSeguinte.col(); j++){
              int linInicio = i * this.strideAltura;
              int colInicio = j * this.strideLargura;
              int linFim = Math.min(linInicio + this.formFiltro[0], entrada.lin());
              int colFim = Math.min(colInicio + this.formFiltro[1], entrada.col());
  
              int[] posicaoMaximo = posicaoMaxima(entrada, linInicio, colInicio, linFim, colFim);
              int rowMaximo = posicaoMaximo[0];
              int colMaximo = posicaoMaximo[1];
  
              double valorGradSeguinte = gradSeguinte.dado(i, j);
              gradEntrada.editar(rowMaximo, colMaximo, valorGradSeguinte);
          }
      }
  }
  
   private int[] posicaoMaxima(Mat matriz, int linInicio, int colInicio, int linFim, int colFim){
      int[] posicaoMaximo = new int[]{linInicio, colInicio};
      double valorMaximo = Double.NEGATIVE_INFINITY;
  
      for(int row = linInicio; row < linFim; row++){
          for(int col = colInicio; col < colFim; col++){
              double valorAtual = matriz.dado(row, col);
              if(valorAtual > valorMaximo){
                  valorMaximo = valorAtual;
                  posicaoMaximo[0] = row;
                  posicaoMaximo[1] = col;
              }
          }
      }
  
      return posicaoMaximo;
  }
 
   @Override
   public int[] formatoEntrada(){
      return this.formEntrada;
   }

   @Override
   public int[] formatoSaida(){
      return this.formSaida;
   }

   @Override
   public int tamanhoSaida(){
      return this.saida.length * this.saida[0].lin() * this.saida[0].col();
   }

   public int[] formatoFiltro(){
      return this.formFiltro;
   }

   public int[] formatoStride(){
      return new int[]{strideLargura, strideAltura};
   }

   @Override
   public int numParametros(){
      return 0;
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
