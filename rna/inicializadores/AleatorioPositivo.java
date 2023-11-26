package rna.inicializadores;

import rna.core.Mat;

public class AleatorioPositivo extends Inicializador{

   /**
    * Inicializa os pesos aleatoriamente dentro do intervalo {@code 0 : x}
    * @param m matriz que será inicializada.
    * @param x valor que será usado como máximo e mínimo na aleatorização.
    */
   @Override
   public void inicializar(Mat m, double x){
      for(int i = 0; i < m.lin; i++){
         for(int j = 0; j < m.col; j++){
            m.editar(i, j, super.random.nextDouble(0, x));
         }
      }
   }

   @Override
   public void configurarSeed(long seed){
      super.configurarSeed(seed);
   }
}
