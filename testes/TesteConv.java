package testes;

import ged.Ged;
import rna.core.Mat;
import rna.estrutura.*;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Xavier;

public class TesteConv{
   static Ged ged = new Ged();

   public static void main(String[] args){
      ged.limparConsole();
      
      double[][] e = {
         {1, 6, 2},
         {5, 3, 1},
         {7, 0, 4},
      };
      
      double[][] f1 = {
         {1, 2},
         {-1, 0},
      };
      
      int[] formatoEntrada = {e.length, e[0].length, 1};
      int[] formatoFiltro = {f1.length, f1[0].length};

      double[][][] entrada = new double[1][][];
      entrada[0] = e;

      // ---------------------------------------------------------
      Inicializador ini = new Xavier();

      Convolucional conv = new Convolucional(formatoEntrada, formatoFiltro, 2, false);
      conv.inicializar(ini, ini, 0);
      conv.filtros[0][0] = new Mat(f1);
      conv.filtros[1][0] = new Mat(f1);

      Flatten flat = new Flatten(conv.formatoSaida());
      
      Densa densa1 = new Densa(conv.tamanhoSaida(), 3);
      densa1.configurarAtivacao("sigmoid");
      densa1.inicializar(ini, ini, 0);
      
      Densa densa2 = new Densa(densa1.tamanhoSaida(), 2);
      densa2.configurarAtivacao("softmax");
      densa2.inicializar(ini, ini, 0);

      ConvNet cnn = new ConvNet();
      cnn.add(conv);
      cnn.add(flat);
      cnn.add(densa1);
      cnn.add(densa2);
      cnn.calcularSaida(entrada);

      ged.imprimirArray(cnn.obterSaida(), "saida cnn");
   }
}


class ConvNet{
   public Camada[] camadas = new Camada[0];

   /**
    * Adiciona camadas ao modelo.
    * @param camada nova camada.
    */
   public void add(Camada camada){
      Camada[] c = this.camadas;
      this.camadas = new Camada[c.length+1];

      for(int i = 0; i < c.length; i++){
         this.camadas[i] = c[i];
      }
      this.camadas[this.camadas.length-1] = camada;
   }

   /**
    * Feedforward
    * @param entrada entrada
    */
   public void calcularSaida(double[][][] entrada){
      this.camadas[0].calcularSaida(entrada);
      for(int i = 1; i < this.camadas.length; i++){
         this.camadas[i].calcularSaida(this.camadas[i-1].obterSaida());
      }
   }

   public double[] obterSaida(){
      double[] saida = (double[]) this.camadas[this.camadas.length-1].obterSaida();
      return  saida;
   }
}
