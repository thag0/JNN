package testes;

import java.text.DecimalFormat;

import rna.estrutura.RedeNeural;
import rna.inicializadores.*;
import rna.avaliacao.perda.*;
import rna.otimizadores.*;
import ged.Dados;
import ged.Ged;

public class ExemploClassificacao{
   
   public static void main(String[] args){
      Ged ged = new Ged();
      ged.limparConsole();

      //carregando dados e tratando
      //removendo linha com nomes das categorias
      //tranformando a ultima coluna em categorização binária
      Dados iris = ged.lerCsv("./dados/iris.csv");
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
      int[] arq = {qEntradas, 9, 9, qSaidas};
      RedeNeural rede = new RedeNeural(arq);

      Perda perda = new EntropiaCruzada();
      Otimizador otimizador = new SGD(0.001, 0.99);
      Inicializador inicializador = new Xavier();

      rede.compilar(perda, otimizador, inicializador);
      rede.configurarAtivacao("sigmoid");
      rede.configurarAtivacao(rede.obterCamadaSaida(), "softmax");
      
      //treinando e avaliando os resultados
      rede.treinar(treinoX, treinoY, 5_000);
      double acurariaRede = rede.avaliador.acuracia(testeX, testeY);
      double perdaRede = rede.avaliador.entropiaCruzada(testeX, testeY);
      System.out.println("Acurácia = " + formatarDecimal(acurariaRede*100, 4) + "%");
      System.out.println("Perda = " + perdaRede);

      int[][] matrizConfusao = rede.avaliador.matrizConfusao(testeX, testeY);
      Dados d = new Dados(matrizConfusao);
      d.editarNome("Matriz de confusão");
      d.imprimir();
   }

   public static void compararSaidaRede(RedeNeural rede, double[][] dadosEntrada, double[][] dadosSaida, String texto){
      int nEntrada = rede.obterTamanhoEntrada();
      int nSaida = rede.obterCamadaSaida().quantidadeNeuronios();

      double[] entrada_rede = new double[nEntrada];
      double[] saida_rede = new double[nSaida];

      System.out.println("\n" + texto);

      //mostrar saída da rede comparada aos dados
      for(int i = 0; i < dadosEntrada.length; i++){
         for(int j = 0; j < dadosEntrada[0].length; j++){
            entrada_rede[j] = dadosEntrada[i][j];
         }

         rede.calcularSaida(entrada_rede);
         saida_rede = rede.obterSaidas();

         //apenas formatação
         if(i < 10) System.out.print("Dado 00" + i + " |");
         else if(i < 100) System.out.print("Dado 0" + i + " |");
         else System.out.print("Dado " + i + " |");
         for(int j = 0; j < entrada_rede.length; j++){
            System.out.print(" " + entrada_rede[j] + " ");
         }

         System.out.print(" - ");
         for(int j = 0; j < dadosSaida[0].length; j++){
            System.out.print(" " + dadosSaida[i][j]);
         }
         System.out.print(" | Rede ->");
         for(int j = 0; j < nSaida; j++){
            System.out.print("  " + formatarDecimal(saida_rede[j], 4));
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
}
