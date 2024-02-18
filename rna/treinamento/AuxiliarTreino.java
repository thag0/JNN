package rna.treinamento;

import java.util.Random;

import rna.avaliacao.perda.Perda;
import rna.camadas.Camada;
import rna.core.OpArray;
import rna.core.OpMatriz;
import rna.core.Tensor4D;

/**
 * Classe auxiliar no treinamento, faz uso de ferramentas que podem
 * ser compartilhadas entre os diferentes tipos de modelos de treinamento.
 */
public class AuxiliarTreino{

   /**
    * Gerador de números aleatórios.
    */
   Random random = new Random();

   /**
    * Operador matricial.
    */
   OpMatriz opmat = new OpMatriz();

   /**
    * Operador para arrays.
    */
   OpArray oparr = new OpArray();

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
    *    Os gradientes iniciais são calculados usando a derivada da função de perda em relação
    *    aos erros do modelo.
    * </p>
    * <p>
    *    A partir disso, são retropropagados de volta da última camada do modelo até a primeira.
    * </p>
    * @param camadas conjunto de camadas de um modelo.
    * @param perda função de perda configurada para o modelo.
    * @param real saída real que será usada para calcular os erros e gradientes.
    */
   public void backpropagation(Camada[] camadas, Perda perda, double[] real){
      Camada saida = camadas[camadas.length-1];
      double[] previsto = saida.saidaParaArray();
      double[] gradPrev = perda.derivada(previsto, real);

      Tensor4D g = new Tensor4D(gradPrev);
      saida.calcularGradiente(g);
      for(int i = camadas.length-2; i >= 0; i--){
         camadas[i].calcularGradiente(camadas[i+1].obterGradEntrada());
      }
   }

   /**
    * Embaralha os dados da matriz usando o algoritmo Fisher-Yates.
    * @param entradas matriz com os dados de entrada.
    * @param saidas matriz com os dados de saída.
    */
   public void embaralharDados(Object[] entradas, Object[] saidas){
      int linhas = entradas.length;
      int i, idAleatorio;

      Object temp;
      for(i = linhas - 1; i > 0; i--){
         idAleatorio = random.nextInt(i+1);

         //trocar entradas
         temp = entradas[i];
         entradas[i] = entradas[idAleatorio];
         entradas[idAleatorio] = temp;

         //trocar saídas
         temp = saidas[i];
         saidas[i] = saidas[idAleatorio];
         saidas[idAleatorio] = temp;
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
   public Object[] obterSubMatriz(Object[] dados, int inicio, int fim){
      if(inicio < 0 || fim > dados.length || inicio >= fim){
         throw new IllegalArgumentException("Índices de início ou fim inválidos.");
      }

      int linhas = fim - inicio;
      Object[] subMatriz = new double[linhas][];

      for(int i = 0; i < linhas; i++){
         subMatriz[i] = dados[i + inicio];
      }

      return subMatriz;
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
   }

   /**
    * Experimental ainda.
    * @param entrada
    * @return
    */
   Object[] entradaParaArray(Object entrada){
      Object[] elementos = new Object[0];

      if(entrada instanceof Object[]){
         elementos = (Object[]) entrada;
      
      }else if(entrada instanceof Tensor4D){
         Tensor4D t = (Tensor4D) entrada;
         int idArray = 0;
         int[] dim = t.dimensoes();

         for(int i = dim.length-1; i >= 0; i--){
            if(dim[i] > 1) idArray = i;
         }

         Tensor4D[] amostras = new Tensor4D[dim[idArray]];

         for(int i = 0; i < amostras.length; i++){
            if(idArray == 0){//tensores 3d
               amostras[i] = new Tensor4D(t.array3D(i));

            }else if(idArray == 1){//matrizes
               amostras[i] = new Tensor4D(t.array2D(0, i));
            
            }else if(idArray == 2){//vetores
               amostras[i] = new Tensor4D(1, 1, 1, dim[idArray+1]);
               for(int j = 0; j < amostras[i].dim3(); j++){
                  amostras[i].copiar(t.array1D(0, 0, i), 0, 0, j);
               }
            
            }else if(idArray == 3){//escalar
               amostras[i] = new Tensor4D(1, 1, 1, 1);
               amostras[i].editar(0, 0, 0, 0, t.elemento(0, 0, 0, i));
            }
         }

         //temp
         throw new UnsupportedOperationException(
            "\nTransformação dos dados de entrada para tensores não testada."
         );
      }else{
         throw new IllegalArgumentException(
            "Tipo de objeto (" + entrada.getClass().getSimpleName() + ") inválido."
         );
      }

      return elementos;
   }
}
