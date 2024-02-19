package testes;

import java.awt.image.BufferedImage;
import java.nio.Buffer;
import java.sql.Time;
import java.util.concurrent.TimeUnit;

import lib.ged.Dados;
import lib.ged.Ged;
import lib.geim.Geim;
import rna.avaliacao.perda.EntropiaCruzada;
import rna.camadas.AvgPooling;
import rna.camadas.Convolucional;
import rna.camadas.Densa;
import rna.camadas.Dropout;
import rna.camadas.Flatten;
import rna.camadas.MaxPooling;
import rna.core.Mat;
import rna.core.OpArray;
import rna.core.OpMatriz;
import rna.core.OpTensor4D;
import rna.core.Tensor4D;
import rna.inicializadores.AleatorioPositivo;
import rna.inicializadores.Inicializador;
import rna.inicializadores.Zeros;
import rna.inicializadores.GlorotUniforme;
import rna.inicializadores.Identidade;
import rna.treinamento.AuxiliarTreino;

@SuppressWarnings("unused")
public class MatrizTeste{
   static Ged ged = new Ged();
   static OpArray oparr = new OpArray();
   static OpMatriz opmat = new OpMatriz();
   static OpTensor4D optensor = new OpTensor4D();
   static Geim geim = new Geim();
   
   public static void main(String[] args){
      ged.limparConsole();
      
      
      Tensor4D tensor = new Tensor4D(1, 2, 2, 2);
      
      tensor.preencherContador(true);
      tensor.print(2);
      
      tensor.reformatar(1, 1, 2, 4);
      tensor.print(2);
   }

   static Object[] teste(Object obj){
      Object[] elementos = new Object[0];

      if(obj instanceof Object[]){
         elementos = (Object[]) obj;
      
      }else if(obj instanceof Tensor4D){
         Tensor4D t = (Tensor4D) obj;
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
               amostras[i] = new Tensor4D(1, 1, 1, dim[idArray]);
               for(int j = 0; j < amostras[i].dim3(); j++){
                  amostras[i].copiar(t.array1D(0, 0, i), 0, 0, j);
               }
            
            }else if(idArray == 3){//escalar
               amostras[i] = new Tensor4D(1, 1, 1, 1);
               amostras[i].editar(0, 0, 0, 0, t.elemento(0, 0, 0, i));
            }
         }

         for(Tensor4D tensor : amostras){
            tensor.print(1);
         }

      }else{
         throw new IllegalArgumentException(
            "Tipo de objeto (" + obj.getClass().getSimpleName() + ") inválido."
         );
      }

      return elementos;
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