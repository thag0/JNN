package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.core.Array;
import rna.core.OpMatriz;
import rna.estrutura.CamadaDensa;

class Auxiliar{
   Random random = new Random();
   OpMatriz mat = new OpMatriz();
   Array arr = new Array();

   /**
    * Configura a seed inicial do gerador de números aleatórios.
    * @param seed nova seed.
    */
   public void configurarSeed(long seed){
      this.random.setSeed(seed);
   }

   /**
    * Calcular os gradientes de todas as camadas da rede de acordo com os valores
    * previstos e a função de perda configurada pela Rede Neural.
    * <p>
    *    Multiplicar os erros pela derivada a função de ativação
    *    da camada ta deixando o treinamento muito mais lento, não 
    *    sei se deveria acontecer isso.
    * </p>
	 * @param redec Lista de camadas densas da Rede Neural.
    * @param perda função de perda da Rede Neural.
    * @param real valores reais dos dados preditos.
    */
   public void calcularGradientes(CamadaDensa[] redec, Perda perda, double[] real){
      //saida
      CamadaDensa saida = redec[redec.length-1];
      double[] previsto = saida.obterSaida().linha(0);
      double[] grads = perda.derivada(previsto, real);
      saida.gradientes.copiar(0, grads);

      //ocultas
      for(int i = redec.length-2; i >= 0; i--){
         CamadaDensa camada = redec[i];

         mat.mult(redec[i+1].gradientes, redec[i+1].pesos.transpor(), camada.gradientes);
         camada.calcularDerivadas();
         mat.hadamard(camada.derivada, camada.gradientes, camada.gradientes);
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
         copiarArray(entradas[i], tempEntradas);
         copiarArray(entradas[idAleatorio], entradas[i]);
         copiarArray(tempEntradas, entradas[idAleatorio]);

         //trocar saídas
         copiarArray(saidas[i], tempSaidas);
         copiarArray(saidas[idAleatorio], saidas[i]);
         copiarArray(tempSaidas, saidas[idAleatorio]); 
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

   /**
    * Adiciona o novo valor de perda no final do histórico.
    * @param historico histórico com os valores de perda da rede.
    * @param valor novo valor que será adicionado.
    */
   double[] adicionarPerda(double[] historico, double valor){
      double[] aux = historico;
      historico = new double[historico.length + 1];
      
      for(int i = 0; i < aux.length; i++){
         historico[i] = aux[i];
      }
      historico[historico.length-1] = valor;

      return historico;
   }

   /**
    * Copia todo o conteúdo do array fornecido para o destino.
    * @param arr array contendo os dados.
    * @param dest destino da cópia.
    */
   void copiarArray(double[] arr, double[] dest){
      if(arr.length != dest.length){
         throw new IllegalArgumentException(
            "Os arrays devem conter o mesmo tamanho"
         );
      }

      System.arraycopy(arr, 0, dest, 0, dest.length);
      // for(int i = 0; i < arr.length; i++){
      //    dest[i] = arr[i];
      // }
   }
}
