package externos.lib.ged;

import java.util.Random;


/**
 * Gerenciador de treino e teste do Ged
 */
class TreinoTeste{ 

   /**
    * Implementações relacionadas a manipulação de dados de treino e teste da rede neural.
    */
   public TreinoTeste() {}

   //embaralhar

   public void embaralharDados(Object dados) {
      if (dados instanceof int[][]) {
         int[][] d = (int[][]) dados;
         embaralharDados(d);
         dados = (Object) d;
      
      } else if (dados instanceof float[][]) {
         float[][] d = (float[][]) dados;
         embaralharDados(d);
         dados = (Object) d;
      
      } else if (dados instanceof double[][]) {
         double[][] d = (double[][]) dados;
         embaralharDados(d);
         dados = (Object) d;
      
      } else {
         throw new IllegalArgumentException("Tipo de dados não suportado.");
      }
   }

   private void embaralharDados(int[][] dados) {
      Random random = new Random();
      int linhas = dados.length;

      int[] temp = new int[dados[0].length];
      int j;
      for (int i = linhas - 1; i > 0; i--) {
         j = random.nextInt(i + 1);

         System.arraycopy(dados[i], 0, temp, 0, temp.length);
         System.arraycopy(dados[j], 0, dados[i], 0, dados[i].length);
         System.arraycopy(temp, 0, dados[j], 0, dados[j].length);
      }
   }

   private void embaralharDados(float[][] dados) {
      Random random = new Random();
      int linhas = dados.length;

      float[] temp = new float[dados[0].length];
      int j;
      for (int i = linhas - 1; i > 0; i--) {
         j = random.nextInt(i + 1);

         System.arraycopy(dados[i], 0, temp, 0, temp.length);
         System.arraycopy(dados[j], 0, dados[i], 0, dados[i].length);
         System.arraycopy(temp, 0, dados[j], 0, dados[j].length);
      }
   }

   private void embaralharDados(double[][] dados) {
      Random random = new Random();
      int linhas = dados.length;

      double[] temp = new double[dados[0].length];
      int j;
      for (int i = linhas - 1; i > 0; i--) {
         j = random.nextInt(i + 1);

         System.arraycopy(dados[i], 0, temp, 0, temp.length);
         System.arraycopy(dados[j], 0, dados[i], 0, dados[i].length);
         System.arraycopy(temp, 0, dados[j], 0, dados[j].length);
      }
   }

   //separar entrada

   public Object separarDadosEntrada(Object dados, int colunas) {
      if (dados instanceof int[][]) {
         int[][] m = (int[][]) dados;
         return separarDadosEntrada(m, colunas);
      
      } else if (dados instanceof float[][]) {
         float[][] m = (float[][]) dados;
         return separarDadosEntrada(m, colunas);
      
      } else if (dados instanceof double[][]) {
         double[][] m = (double[][]) dados;
         return separarDadosEntrada(m, colunas);
      
      } else {
         throw new IllegalArgumentException(
            "Tipo de dado (" + dados.getClass().getSimpleName() +") não suportado."
         );
      }  
   }

   private int[][] separarDadosEntrada(int[][] dados, int colunas) {
      if (colunas > dados[0].length) {
         throw new IllegalArgumentException("O número de colunas fornecido é maior do que o número de colunas disponíveis nos dados.");
      }
      if (colunas < 1) {
         throw new IllegalArgumentException("A quantidade de colunas extraídas não pode ser menor que um");
      }

      int[][] dadosEntrada = new int[dados.length][colunas];
      for (int i = 0; i < dadosEntrada.length; i++) {
         System.arraycopy(dados[i], 0, dadosEntrada[i], 0, colunas);
      }
      
      return dadosEntrada;
   }

   private float[][] separarDadosEntrada(float[][] dados, int colunas) {
      if (colunas > dados[0].length) {
         throw new IllegalArgumentException("O número de colunas fornecido é maior do que o número de colunas disponíveis nos dados.");
      }
      if (colunas < 1) {
         throw new IllegalArgumentException("A quantidade de colunas extraídas não pode ser menor que um");
      }

      float[][] dadosEntrada = new float[dados.length][colunas];
      for (int i = 0; i < dadosEntrada.length; i++) {
         System.arraycopy(dados[i], 0, dadosEntrada[i], 0, colunas);
      }
      
      return dadosEntrada;
   }

   private double[][] separarDadosEntrada(double[][] dados, int colunas) {
      if (colunas > dados[0].length) {
         throw new IllegalArgumentException("O número de colunas fornecido é maior do que o número de colunas disponíveis nos dados.");
      }
      if (colunas < 1) {
         throw new IllegalArgumentException("A quantidade de colunas extraídas não pode ser menor que um");
      }

      double[][] dadosEntrada = new double[dados.length][colunas];
      for (int i = 0; i < dadosEntrada.length; i++) {
         System.arraycopy(dados[i], 0, dadosEntrada[i], 0, colunas);
      }
      
      return dadosEntrada;
   }

   //separar saida

   public Object separarDadosSaida(Object dados, int colunas) {
      if (dados instanceof int[][]) {
         int[][] m = (int[][]) dados;
         return separarDadosSaida(m, colunas);
      
      } else if (dados instanceof float[][]) {
         float[][] m = (float[][]) dados;
         return separarDadosSaida(m, colunas);
      
      } else if (dados instanceof double[][]) {
         double[][] m = (double[][]) dados;
         return separarDadosSaida(m, colunas);
      
      } else {
         throw new IllegalArgumentException(
            "Tipo de dado (" + dados.getClass().getSimpleName() +") não suportado."
         );
      }  
   }

   private int[][] separarDadosSaida(int[][] dados, int colunas) {
      if (colunas > dados[0].length) {
         throw new IllegalArgumentException("O número de colunas fornecido é maior do que o número de colunas disponíveis nos dados.");
      }
      if (colunas < 1) {
         throw new IllegalArgumentException("A quantidade de colunas extraídas não pode ser menor que um");
      }

      int[][] dadosSaida = new int[dados.length][colunas];
      int indiceInicial = dados[0].length - colunas;

      for (int i = 0; i < dados.length; i++) {
         System.arraycopy(dados[i], indiceInicial, dadosSaida[i], 0, colunas);
      }

      return dadosSaida;
   }

   private float[][] separarDadosSaida(float[][] dados, int colunas) {
      if (colunas > dados[0].length) {
         throw new IllegalArgumentException("O número de colunas fornecido é maior do que o número de colunas disponíveis nos dados.");
      }
      if (colunas < 1) {
         throw new IllegalArgumentException("A quantidade de colunas extraídas não pode ser menor que um");
      }

      float[][] dadosSaida = new float[dados.length][colunas];
      int indiceInicial = dados[0].length - colunas;

      for (int i = 0; i < dados.length; i++) {
         System.arraycopy(dados[i], indiceInicial, dadosSaida[i], 0, colunas);
      }

      return dadosSaida;
   }

   private double[][] separarDadosSaida(double[][] dados, int colunas) {
      if (colunas > dados[0].length) {
         throw new IllegalArgumentException("O número de colunas fornecido é maior do que o número de colunas disponíveis nos dados.");
      }
      if (colunas < 1) {
         throw new IllegalArgumentException("A quantidade de colunas extraídas não pode ser menor que um");
      }

      double[][] dadosSaida = new double[dados.length][colunas];
      int indiceInicial = dados[0].length - colunas;

      for (int i = 0; i < dados.length; i++) {
         System.arraycopy(dados[i], indiceInicial, dadosSaida[i], 0, colunas);
      }

      return dadosSaida;
   }

   //separar treino teste

   public Object separarTreinoTeste(Object dados, float tamanhoTeste) {
      if (dados instanceof int[][]) {
         int[][] m = (int[][]) dados;
         return separarTreinoTesteI(m, tamanhoTeste);
      
      } else if (dados instanceof float[][]) {
         float[][] m = (float[][]) dados;
         return separarTreinoTesteF(m, tamanhoTeste);
      
      } else if (dados instanceof double[][]) {
         double[][] m = (double[][]) dados;
         return separarTreinoTesteD(m, tamanhoTeste);
      
      } else {
         throw new IllegalArgumentException(
            "Tipo de dado (" + dados.getClass().getSimpleName() +") não suportado."
         );
      }  
   }

   private int[][][] separarTreinoTesteI(int[][] dados, float tamanhoTeste) {
      if (dados == null) throw new IllegalArgumentException("O conjunto de dados é nulo.");
      if (tamanhoTeste < 0 || tamanhoTeste > 1) {
         throw new IllegalArgumentException("O tamanho dos dados de teste deve ser maior que zero e menor que um.");
      }

      int linhasTeste = (int) (dados.length*tamanhoTeste);
      int linhasTreino = dados.length - linhasTeste;
      int colunas = dados[0].length;

      int[][] treino = new int[linhasTreino][colunas];
      int[][] teste = new int[linhasTeste][colunas];

      //método nativo, parece ser mais rápido que uma cópia manual
      System.arraycopy(dados, 0, treino, 0, linhasTreino);// copiar treino
      System.arraycopy(dados, linhasTreino, teste, 0, linhasTeste);// copiar teste

      return new int[][][]{treino, teste};
   }

   private float[][][] separarTreinoTesteF(float[][] dados, float tamanhoTeste) {
      if (dados == null) throw new IllegalArgumentException("O conjunto de dados é nulo.");
      if (tamanhoTeste < 0 || tamanhoTeste > 1) {
         throw new IllegalArgumentException("O tamanho dos dados de teste deve ser maior que zero e menor que um.");
      }

      int linhasTeste = (int) (dados.length*tamanhoTeste);
      int linhasTreino = dados.length - linhasTeste;
      int colunas = dados[0].length;

      float[][] treino = new float[linhasTreino][colunas];
      float[][] teste = new float[linhasTeste][colunas];

      //método nativo, parece ser mais rápido que uma cópia manual
      System.arraycopy(dados, 0, treino, 0, linhasTreino);// copiar treino
      System.arraycopy(dados, linhasTreino, teste, 0, linhasTeste);// copiar teste

      return new float[][][]{treino, teste};
   }

   private double[][][] separarTreinoTesteD(double[][] dados, float tamanhoTeste) {
      if (dados == null) throw new IllegalArgumentException("O conjunto de dados é nulo.");
      if (tamanhoTeste < 0 || tamanhoTeste > 1) {
         throw new IllegalArgumentException("O tamanho dos dados de teste deve ser maior que zero e menor que um.");
      }

      int linhasTeste = (int) (dados.length*tamanhoTeste);
      int linhasTreino = dados.length - linhasTeste;
      int colunas = dados[0].length;

      double[][] treino = new double[linhasTreino][colunas];
      double[][] teste = new double[linhasTeste][colunas];

      //método nativo, parece ser mais rápido que uma cópia manual
      System.arraycopy(dados, 0, treino, 0, linhasTreino);// copiar treino
      System.arraycopy(dados, linhasTreino, teste, 0, linhasTeste);// copiar teste

      return new double[][][]{treino, teste};
   }
}
