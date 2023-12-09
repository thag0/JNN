package testes;

import java.io.Serializable;

import ged.Dados;
import ged.Ged;
import rna.ativacoes.Sigmoid;
import rna.modelos.RedeNeural;
import rna.serializacao.Serializador;

@SuppressWarnings("unused")
public class SalvandoRede{
   public static void main(String[] args) {
      Ged ged = new Ged();
      ged.limparConsole();

      Dados xor = ged.lerCsv("./dados/xor.csv");
      double[][] dados = ged.dadosParaDouble(xor);

      double[][] entrada = (double[][]) ged.separarDadosEntrada(dados, 2);
      double[][] saida   = (double[][]) ged.separarDadosSaida(dados, 1);

      RedeNeural rede = criar();
      // rede.treinar(entrada, saida, 10_000);
      rede.diferencaFinita(entrada, saida, 0.1, 0.1, 30_000);
      // Serializador.salvar(rede, "/rede-xor.txt");

      System.out.println("Perda: " + rede.avaliador.erroMedioQuadrado(entrada, saida));

      for(int i = 0; i < 2; i++){
         for(int j = 0; j < 2; j++){
            double[] e = {i, j};
            rede.calcularSaida(e);
            System.out.println(i + " - " + j + " = " + rede.saidaParaArray()[0]);
         }
      }
   }

   static RedeNeural criar(){
      RedeNeural rede = new RedeNeural(new int[]{2, 3, 1});
      rede.compilar();
      rede.configurarAtivacao(new Sigmoid());
      return rede;
   }

   static RedeNeural ler(String caminho){
      return Serializador.ler(caminho);
   }
}
