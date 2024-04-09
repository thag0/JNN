package testes.modelos;

import java.io.Serializable;

import rna.camadas.Convolucional;
import lib.ged.Dados;
import lib.ged.Ged;
import rna.ativacoes.Sigmoid;
import rna.avaliacao.perda.MSE;
import rna.camadas.Camada;
import rna.camadas.Densa;
import rna.inicializadores.GlorotUniforme;
import rna.inicializadores.Zeros;
import rna.modelos.*;
import rna.otimizadores.*;
import rna.serializacao.Serializador;

@SuppressWarnings("unused")
public class SalvandoRede{
   public static void main(String[] args){
      Ged ged = new Ged();
      ged.limparConsole();
      String caminho = "./modelo-teste.nn";
      Serializador serializador = new Serializador();
      
      // Sequencial modelo = ler(caminho);
      Sequencial modelo = criar();

      modelo.info();

      serializador.salvar(modelo, caminho);
   }

   static Sequencial criar(){
      Sequencial modelo = new Sequencial(new Camada[]{
         new Convolucional(new int[]{2, 10, 10}, new int[]{3, 3}, 2),
      });

      modelo.compilar("sgd", "mse");
      return modelo;
   }

   static Sequencial ler(String caminho){
      return new Serializador().lerSequencial(caminho); 
   }
}
