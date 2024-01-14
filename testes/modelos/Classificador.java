package testes.modelos;

import java.text.DecimalFormat;

import rna.inicializadores.*;
import rna.modelos.Modelo;
import rna.modelos.RedeNeural;
import rna.modelos.Sequencial;
import rna.avaliacao.perda.*;
import rna.camadas.Camada;
import rna.camadas.Densa;
import rna.camadas.Dropout;
import rna.otimizadores.*;
import lib.ged.Dados;
import lib.ged.Ged;

public class Classificador{
   
   public static void main(String[] args){
      Ged ged = new Ged();
      ged.limparConsole();

      //carregando dados e tratando
      //removendo linha com nomes das categorias
      //tranformando a ultima coluna em categorização binária
      Dados iris = ged.lerCsv("./dados/csv/iris.csv");
      ged.removerLinha(iris, 0);
      int[] shape = ged.shapeDados(iris);
      int ultimoIndice = shape[1]-1;
      ged.categorizar(iris, ultimoIndice);
      System.out.println("Tamanho dados = " + iris.shapeInfo());

      //separando dados de treino e teste
      double[][] dados = ged.dadosParaDouble(iris);
      ged.embaralharDados(dados);
      double[][][] treinoTeste = (double[][][]) ged.separarTreinoTeste(dados, 0.25f);
      double[][] treino = treinoTeste[0];
      double[][] teste = treinoTeste[1];
      int qEntradas = 4;// dados de entrada (features)
      int qSaidas = 3;// classificações (class)

      double[][] treinoX = (double[][]) ged.separarDadosEntrada(treino, qEntradas);
      double[][] treinoY = (double[][]) ged.separarDadosSaida(treino, qSaidas);

      double[][] testeX = (double[][]) ged.separarDadosEntrada(teste, qEntradas);
      double[][] testeY = (double[][]) ged.separarDadosSaida(teste, qSaidas);

      //criando e configurando a rede neural
      Sequencial modelo = new Sequencial(new Camada[]{
         new Densa(qEntradas, 10, "sigmoid"),
         new Dropout(0.3),
         new Densa(10, "sigmoid"),
         new Dropout(0.3),
         new Densa(qSaidas, "softmax")
      });

      modelo.compilar(
         new SGD(0.001, 0.99),
         new EntropiaCruzada(),
         new Xavier()
      );
      modelo.configurarHistorico(true);
      modelo.info();
      
      //treinando e avaliando os resultados
      modelo.treinar(treinoX, treinoY, 2_000, false);
      double acc = modelo.avaliador.acuracia(testeX, testeY);
      System.out.println("Acurácia = " + formatarDecimal(acc*100, 4) + "%");
      System.out.println("Perda = " + modelo.avaliar(testeX, testeY));

      int[][] matrizConfusao = modelo.avaliador.matrizConfusao(testeX, testeY);
      Dados d = new Dados(matrizConfusao);
      d.editarNome("Matriz de confusão");
      d.imprimir();

      exportarHistoricoPerda(modelo, ged);
      // compararSaidaRede(rede, testeX, testeY, "");
   }

   public static void compararSaidaRede(RedeNeural rede, double[][] dadosEntrada, double[][] dadosSaida, String texto){
      int nEntrada = rede.obterTamanhoEntrada();
      int nSaida = rede.camadaSaida().numNeuronios();

      double[] entradaRede = new double[nEntrada];
      double[] saidaRede = new double[nSaida];

      System.out.println("\n" + texto);

      //mostrar saída da rede comparada aos dados
      for(int i = 0; i < dadosEntrada.length; i++){
         for(int j = 0; j < dadosEntrada[0].length; j++){
            entradaRede[j] = dadosEntrada[i][j];
         }

         rede.calcularSaida(entradaRede);
         saidaRede = rede.saidaParaArray();

         //apenas formatação
         if(i < 10) System.out.print("Dado 00" + i + " |");
         else if(i < 100) System.out.print("Dado 0" + i + " |");
         else System.out.print("Dado " + i + " |");
         for(int j = 0; j < entradaRede.length; j++){
            System.out.print(" " + entradaRede[j] + " ");
         }

         System.out.print(" - ");
         for(int j = 0; j < dadosSaida[0].length; j++){
            System.out.print(" " + dadosSaida[i][j]);
         }
         System.out.print(" | Rede ->");
         for(int j = 0; j < nSaida; j++){
            System.out.print("  " + formatarDecimal(saidaRede[j], 4));
         }
         System.out.println();
      }
   }

   public static String formatarDecimal(double valor, int casas){
      String valorFormatado = "";

      String formato = "#.";
      for(int i = 0; i < casas; i++) formato += "#";

      DecimalFormat df = new DecimalFormat(formato);
      valorFormatado = df.format(valor);

      return valorFormatado;
   }

   public static void exportarHistoricoPerda(Modelo rede, Ged ged){
      System.out.println("Exportando histórico de perda");
      double[] perdas = rede.historico();
      double[][] dadosPerdas = new double[perdas.length][1];

      for(int i = 0; i < dadosPerdas.length; i++){
         dadosPerdas[i][0] = perdas[i];
      }

      Dados dados = new Dados(dadosPerdas);
      ged.exportarCsv(dados, "historico-perda");
   }
}
