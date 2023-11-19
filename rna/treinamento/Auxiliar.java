package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.core.Matriz;
import rna.estrutura.CamadaDensa;

class Auxiliar{
   Random random = new Random();

   /**
    * Configura a seed inicial do gerador de números aleatórios.
    * @param seed nova seed.
    */
   public void configurarSeed(long seed){
      this.random.setSeed(seed);
   }

   /**
    * 
    * @param camadas
    * @param perda
    * @param real
    */
   public void calcularErros(CamadaDensa[] camadas, Perda perda, double[] real){
      //saida
      CamadaDensa saida = camadas[camadas.length-1];
      double[] erros = perda.derivada(saida.obterSaida()[0], real);
      for(int i = 0; i < saida.tamanhoSaida(); i++){
         saida.erros[0][i] = erros[i];
      }

      //ocultas
      for(int i = camadas.length-2; i >= 0; i--){
         camadas[i].calcularDerivadas();

         double[][] pesoTransposto = Matriz.transpor(camadas[i+1].pesos);
         Matriz.mult(camadas[i+1].erros, pesoTransposto, camadas[i].erros);
         Matriz.hadamard(camadas[i].derivada, camadas[i].erros, camadas[i].erros);
      }
   }
   
   /**
    * Embaralha os dados da matriz usando o algoritmo Fisher-Yates.
    * @param entradas matriz com os dados de entrada.
    * @param saidas matriz com os dados de saída.
    */
   void embaralharDados(double[][] entradas, double[][] saidas){
      int linhas = entradas.length;
      int colEntrada = entradas[0].length;
      int colSaida = saidas[0].length;
  
      //evitar muitas inicializações
      double tempEntradas[] = new double[colEntrada];
      double tempSaidas[] = new double[colSaida];
      int i, idAleatorio;

      for(i = linhas - 1; i > 0; i--){
         idAleatorio = random.nextInt(i+1);

         //trocar entradas
         System.arraycopy(entradas[i], 0, tempEntradas, 0, colEntrada);
         System.arraycopy(entradas[idAleatorio], 0, entradas[i], 0, colEntrada);
         System.arraycopy(tempEntradas, 0, entradas[idAleatorio], 0, colEntrada);

         //trocar saídas
         System.arraycopy(saidas[i], 0, tempSaidas, 0, colSaida);
         System.arraycopy(saidas[idAleatorio], 0, saidas[i], 0, colSaida);
         System.arraycopy(tempSaidas, 0, saidas[idAleatorio], 0, colSaida); 
      }
   }

   /**
    * Dedicado para treino em lote e multithread em implementações futuras.
    * @param dados conjunto de dados completo.
    * @param inicio índice de inicio do lote.
    * @param fim índice final do lote.
    * @return lote contendo os dados de acordo com os índices fornecidos.
    */
   double[][] obterSubMatriz(double[][] dados, int inicio, int fim){
      if(inicio < 0 || fim > dados.length || inicio >= fim){
         throw new IllegalArgumentException("Índices de início ou fim inválidos.");
      }

      int linhas = fim - inicio;
      int colunas = dados[0].length;
      double[][] subMatriz = new double[linhas][colunas];

      for(int i = 0; i < linhas; i++){
         System.arraycopy(dados[inicio+i], 0, subMatriz[i], 0, colunas);
      }

      return subMatriz;
   }
}
