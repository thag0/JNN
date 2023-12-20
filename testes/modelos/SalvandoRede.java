package testes.modelos;

import java.io.Serializable;

import ged.Dados;
import ged.Ged;
import rna.ativacoes.Sigmoid;
import rna.avaliacao.perda.ErroMedioQuadrado;
import rna.estrutura.Camada;
import rna.estrutura.Densa;
import rna.inicializadores.Xavier;
import rna.modelos.*;
import rna.otimizadores.*;
import rna.serializacao.Serializador;

@SuppressWarnings("unused")
public class SalvandoRede{
   public static void main(String[] args) {
      Ged ged = new Ged();
      ged.limparConsole();
      Serializador serializador = new Serializador();
      String caminho = "./rede-xor.txt";

      Dados xor = ged.lerCsv("./dados/xor.csv");
      double[][] dados = ged.dadosParaDouble(xor);

      double[][] entrada = (double[][]) ged.separarDadosEntrada(dados, 2);
      double[][] saida   = (double[][]) ged.separarDadosSaida(dados, 1);

      Sequencial modelo = serializador.lerSequencial(caminho);
      // Sequencial modelo = criarSeq();

      // modelo.treinar(entrada, saida, 10_000);
      // serializador.salvar(modelo, caminho);

      System.out.println("Perda: " + modelo.avaliador.erroMedioQuadrado(entrada, saida));

      for(int i = 0; i < 2; i++){
         for(int j = 0; j < 2; j++){
            double[] e = {i, j};
            modelo.calcularSaida(e);
            System.out.println(i + " - " + j + " = " + modelo.saidaParaArray()[0]);
         }
      }
   }

   static Sequencial criarSeq(){
      Sequencial seq = new Sequencial(new Camada[]{
         new Densa(2, 3, "sigmoid"),
         new Densa(1, "sigmoid")
      });

      seq.compilar(
         new SGD(0.01, 0.95),
         new ErroMedioQuadrado(),
         new Xavier()
      );

      return seq;
   }

   static RedeNeural criarRna(){
      RedeNeural rede = new RedeNeural(new int[]{2, 3, 1});
      rede.compilar(new SGD(), new ErroMedioQuadrado(), new Xavier());
      rede.configurarAtivacao("sigmoid");
      return rede;
   }

}
