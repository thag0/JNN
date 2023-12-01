package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.core.Array;
import rna.core.OpMatriz;
import rna.estrutura.Densa;

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
    * Realiza a retropropagação de gradientes de cada camada para a atualização de pesos.
    * <p>
    *    Os gradientes iniciais são calculados usando a derivada da função de perda, com eles
    *    calculados, são retropropagados da última a primeira camada da rede.
    * </p>
    * @param redec conjunto de camadas densas da Rede Neural.
    * @param perda função de perda configurada para a Rede Neural.
    * @param real saída real que será usada para calcular os erros e gradientes.
    */
   public void backpropagation(Densa[] redec, Perda perda, double[] real){
      Densa saida = redec[redec.length-1];
      double[] previsto = saida.obterSaida().linha(0);
      double[] gradPrev = perda.derivada(previsto, real);
      
      saida.calcularGradiente(gradPrev);
      for(int i = redec.length-2; i >= 0; i--){
         redec[i].calcularGradiente(redec[i+1].gradienteEntrada.linha(0));
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
         copiarArray(dados[inicio+1], subMatriz[i]);
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
