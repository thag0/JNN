package rna.inicializadores;

import java.util.Random;

import rna.core.Mat;

/**
 * Classe responsável pelas funções de inicialização dos pesos
 * da Rede Neural.
 */
public abstract class Inicializador{

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

   /**
    * Inicializa os valores do array de acordo com o inicializador configurado.
    * @param m matriz de dados.
    * @param x valor usado pelos inicializadores.
    */
   public abstract void inicializar(Mat m, double x);
}
