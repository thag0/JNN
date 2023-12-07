package rna.modelos;

import rna.avaliacao.perda.Perda;
import rna.core.Mat;
import rna.estrutura.Camada;
import rna.inicializadores.Inicializador;
import rna.otimizadores.Otimizador;

/**
 * Modelo básico ainda.
 */
public class Sequencial{
   public Camada[] camadas = new Camada[0];

   public Sequencial(){

   }

   public Sequencial(Camada[] camadas){
      this.camadas = camadas;
   }

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

   public void inicializar(Inicializador ini){
      if(this.camadas[0].inicializada == false){
         throw new IllegalArgumentException(
            "É necessário que a primeira camada seja inicializada manualmente."
         );
      }

      for(int i = 1; i < this.camadas.length; i++){
         this.camadas[i].inicializar(ini, ini, 0);
         this.camadas[i].configurarId(i);
      }
   }

   /**
    * Feedforward
    * @param entrada entrada.
    */
   public void calcularSaida(Object entrada){
      this.camadas[0].calcularSaida(entrada);
      for(int i = 1; i < this.camadas.length; i++){
         this.camadas[i].calcularSaida(this.camadas[i-1].obterSaida());
      }
   }

   public void treinar(double[][][][] entrada, double[][] saida, int epochs, Perda perda, Otimizador otm, boolean logs){
      otm.inicializar(this.camadas);

      for(int e = 0; e < epochs; e++){
         double perdaEpoca = 0;
         for(int i = 0; i < entrada.length; i++){
            this.calcularSaida(entrada[i]);
            perdaEpoca += perda.calcular(this.obterSaida(), saida[i]);

            double[] g = perda.derivada(this.obterSaida(), saida[i]);
            Mat gradientes = new Mat(g);
            this.calcularGradientes(gradientes);
            otm.atualizar(this.camadas);
         }

         perdaEpoca /= entrada.length;

         if(logs & (e % 100 == 0)){
            System.out.println("Perda (" + e + "): " + perdaEpoca);
         }
      }
   }

   public void calcularGradientes(Object gradSeguinte){
      Camada saida = this.camadas[this.camadas.length-1];
      saida.calcularGradiente(gradSeguinte);

      for(int i = this.camadas.length-2; i >= 0; i--){
         this.camadas[i].calcularGradiente(this.camadas[i+1].obterGradEntrada());
      }
   }

   public double[] obterSaida(){
      double[] saida = (double[]) this.camadas[this.camadas.length-1].obterSaida();
      return  saida;
   }
}
