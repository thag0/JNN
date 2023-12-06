package rna.estrutura;

import rna.avaliacao.perda.Perda;
import rna.core.Mat;
import rna.otimizadores.Otimizador;

public class ConvNet{
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

   public void treinar(double[][][] entrada, double[] real, int epochs, Perda perda, Otimizador otm){
      otm.inicializar(this.camadas);

      for(int i = 0; i < epochs; i++){
         this.calcularSaida(entrada);
         
         double[] g = perda.derivada(this.obterSaida(), real);
         Mat gradientes = new Mat(g);
         this.calcularGradientes(gradientes);
         otm.atualizar(this.camadas);
         
         if(i % 100 == 0){
            System.out.println("Perda: " + perda.calcular(this.obterSaida(), real));
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
