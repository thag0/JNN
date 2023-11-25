package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.core.Array;
import rna.core.Mat;
import rna.core.Matriz;
import rna.estrutura.CamadaDensa;

class Auxiliar{
   Random random = new Random();
   Matriz mat = new Matriz();
   Array arr = new Array();

   /**
    * Configura a seed inicial do gerador de números aleatórios.
    * @param seed nova seed.
    */
   public void configurarSeed(long seed){
      this.random.setSeed(seed);
   }

   /**
    * Calcular os erros de todas as camadas da rede de acordo com os valores
    * previstos e a função de perda configurada pela Rede Neural.
    * <p>
    *    Os erros também podem ser chamados diretamentes de gradientes, os gradientes
    *    específicos da camada são os gradientes em relação aos pesos dela.
    * </p>
	 * @param redec Lista de camadas densas da Rede Neural.
    * @param perda função de perda da Rede Neural.
    * @param real valores reais dos dados preditos.
    */
   public void calcularErros(CamadaDensa[] redec, Perda perda, double[] real){
      //saida
      //multiplicar os erros pela derivada a função de ativação
      //da camada ta deixando o treinamento muito mais lento.
      CamadaDensa saida = redec[redec.length-1];
      double[] erros = perda.derivada(saida.obterSaida().linha(0), real);
      for(int i = 0; i < saida.tamanhoSaida(); i++){
         saida.erros.editar(0, i, erros[i]);
      }

      //ocultas
      for(int i = redec.length-2; i >= 0; i--){
         redec[i].calcularDerivadas();

         Mat pesoT = mat.transpor(redec[i+1].pesos);
         mat.mult(redec[i+1].erros, pesoT, redec[i].erros);
         mat.hadamard(redec[i].derivada, redec[i].erros, redec[i].erros);
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
}
