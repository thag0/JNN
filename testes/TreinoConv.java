package testes;

import lib.ged.Ged;
import rna.camadas.Convolucional;
import rna.camadas.Flatten;
import rna.core.Tensor4D;
import rna.modelos.Sequencial;
import rna.otimizadores.SGD;

public class TreinoConv{
   public static void main(String[] args){
      Ged ged = new Ged();
      ged.limparConsole();

      int[] formEntrada = {1, 10, 10};
      int[] formFiltro = {3, 3};

      Convolucional conv = new Convolucional(formEntrada, formFiltro, 3, "leaky-relu");
      Flatten flat = new Flatten();
      
      Sequencial modelo = new Sequencial();
      modelo.add(conv);
      modelo.add(new Convolucional(new int[]{3, 3}, 2, "leaky-relu"));
      modelo.add(flat);
      modelo.compilar(new SGD(0.01), "entropia-cruzada");
      modelo.info();

      Tensor4D amostra = new Tensor4D(2, formEntrada[0], formEntrada[1], formEntrada[2]);
      amostra.map((x) -> Math.random());

      double[][] saida = new double[2][flat.saida.tamanho()];
      for(int i = 0; i < saida.length; i++){
         for(int j = 0; j < saida[i].length; j++){
            saida[i][j] = Math.random();
         }
      }

      System.out.println("Perda: " + modelo.avaliar(amostra, saida));
      modelo.treinar(amostra, saida, 5*1000, false);

      System.out.println("Perda: " + modelo.avaliar(amostra, saida));
   }

   public static void executarComando(String comando){
      try{
         new ProcessBuilder("cmd", "/c", comando).inheritIO().start().waitFor();
      }catch(Exception e){
         e.printStackTrace();
      }
   }
}
