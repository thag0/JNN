package testes;

import java.util.concurrent.TimeUnit;

import ged.Ged;
import rna.core.Array;
import rna.core.MatrizT;
import rna.inicializadores.Inicializador;

public class Multithread {
   public static void main(String[] args){
      Ged ged = new Ged();
      ged.limparConsole();
      
      Teste t = new Teste(3, 4);
      for(int i = 0; i < t.pesos.length; i++){
         for(int j = 0; j < t.pesos[i].length; j++){
            t.pesos[i][j] = (i* t.pesos.length) + j + 1;
         }
      }
      ged.matIdentidade(t.entrada);

      //treinar e marcar tempo
      long t1, t2;
      long minutos, segundos;

      t1 = System.nanoTime();
      t.calcularSaida(new double[]{1, 1, 1});
      t2 = System.nanoTime();
      
      long tempoDecorrido = t2 - t1;
      long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempoDecorrido);
      minutos = (segundosTotais % 3600) / 60;
      segundos = segundosTotais % 60;
      System.out.println("Concluído em: " + minutos + "m " + segundos + "s");

      ged.imprimirMatriz(t.somatorio);
   }

}

class Teste{
   MatrizT mat = new MatrizT(2);
   public double[][] pesos;
   public double[][] bias;
   boolean usarBias = true;

   public double[][] entrada;
   public double[][] somatorio;
   public double[][] saida;
   public double[][] erros;
   public double[][] gradientes;
   public double[][] gradientesAcumulados;
   public double[][] derivada;

   public Teste(int entrada, int neuronios, boolean usarBias){
      this.usarBias = usarBias;

      this.entrada = new double[1][entrada];
      this.pesos =   new double[entrada][neuronios];
      this.saida =   new double[1][neuronios];
      
      if(usarBias){
         this.bias = new double[saida.length][saida[0].length];
      }

      this.somatorio =              new double[this.saida.length][this.saida[0].length];
      this.derivada =               new double[this.saida.length][this.saida[0].length];
      this.erros =                  new double[this.saida.length][this.saida[0].length];

      this.gradientes =             new double[this.pesos.length][this.pesos[0].length];
      this.gradientesAcumulados =   new double[this.pesos.length][this.pesos[0].length];
   }

   public Teste(int entrada, int neuronios){
      this(entrada, neuronios, true);
   }

   public void inicializar(Inicializador inicializador, double alcance){
      if(inicializador == null){
         throw new IllegalArgumentException(
            "O inicializador não pode ser nulo."
         );
      }

      inicializador.inicializar(this.pesos, alcance);
      if(this.usarBias){
         inicializador.inicializar(this.bias, alcance);
      }
   }

   public void calcularSaida(double[] entrada){
      if(entrada.length != this.tamanhoEntrada()){
         throw new IllegalArgumentException(
            "Entradas (" + entrada.length + 
            ") incompatíveis com a entrada da camada (" + this.tamanhoEntrada() + 
            ")."
         );
      }

      Array.copiar(entrada, this.entrada[0]);     

      mat.mult(this.entrada, this.pesos, this.somatorio);
      if(usarBias){
         mat.add(this.somatorio, this.bias, this.somatorio);
      }
   }

   public int tamanhoEntrada(){
      return this.entrada[0].length;
   }
}