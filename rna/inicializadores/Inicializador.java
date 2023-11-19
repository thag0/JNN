package rna.inicializadores;

import java.util.Random;

/**
 * Classe responsável pelas funções de inicialização dos pesos
 * da Rede Neural.
 */
public class Inicializador{

   /**
    * Gerador de números pseudo aleatórios compartilhado
    * para as classes filhas.
    */
   protected Random random = new Random();

   /**
    * Configura o início do gerador aleatório.
    * @param seed nova seed de início.
    */
   public void configurarSeed(long seed){
      this.random.setSeed(seed);
   }

   public double gerarDouble(){
      return this.random.nextDouble();
   }

   /**
    * Inicializa os valores do array de acordo com o inicializador configurado.
    * @param array array de pesos do neurônio.
    * @param alcance valor de alcance da aleatorização
    */
   public void inicializar(double[][] m, double alcance){
      throw new UnsupportedOperationException(
         "Método de inicialização não implementado."
      );
   }
}
