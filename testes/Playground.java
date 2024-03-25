package testes;

import java.awt.image.BufferedImage;
import java.io.Serial;
import java.nio.Buffer;
import java.sql.Time;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import lib.ged.Dados;
import lib.ged.Ged;
import lib.geim.Geim;
import rna.ativacoes.Argmax;
import rna.ativacoes.Ativacao;
import rna.ativacoes.Softmax;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.avaliacao.perda.EntropiaCruzadaBinaria;
import rna.camadas.AvgPooling;
import rna.camadas.Camada;
import rna.camadas.Convolucional;
import rna.camadas.Densa;
import rna.camadas.Dropout;
import rna.camadas.Entrada;
import rna.camadas.Flatten;
import rna.camadas.MaxPooling;
import rna.core.Mat;
import rna.core.OpArray;
import rna.core.OpMatriz;
import rna.core.OpTensor4D;
import rna.core.Tensor4D;
import rna.core.Utils;
import rna.inicializadores.AleatorioPositivo;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Zeros;
import rna.modelos.RedeNeural;
import rna.modelos.Sequencial;
import rna.otimizadores.SGD;
import rna.serializacao.Serializador;
import rna.inicializadores.GlorotUniforme;
import rna.inicializadores.Identidade;
import rna.treinamento.AuxiliarTreino;

@SuppressWarnings("unused")
public class Playground{
   static Ged ged = new Ged();
   static OpArray oparr = new OpArray();
   static OpMatriz opmat = new OpMatriz();
   static OpTensor4D optensor = new OpTensor4D();
   static Geim geim = new Geim();
   static Utils utils = new Utils();
   
   public static void main(String[] args){
      ged.limparConsole();

      var tensor = new Tensor4D(2, 2);
      tensor.preencherContador(true);
      tensor.reformatar(1, 1, 4, 1);
      tensor.print(0);
   }

   /**
    * Mede o tempo de execução da função fornecida.
    * @param func função.
    * @return tempo em nanosegundos.
    */
   static long medirTempo(Runnable func){
      long t = System.nanoTime();
      func.run();
      return System.nanoTime() - t;
   }

}