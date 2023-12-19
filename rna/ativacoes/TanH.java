package rna.ativacoes;

import rna.estrutura.Convolucional;
import rna.estrutura.Densa;

public class TanH extends Ativacao{

   public TanH(){
      super.construir(this::tanh, null);
   }

   @Override
   public void calcular(Densa camada){
      super.aplicarFx(camada.somatorio, camada.saida);
   }

   @Override
   public void derivada(Densa camada){
      //forma manual pra aproveitar os valores pre calculados
      double grad, d;
      int i, j;
      int linhas = camada.saida.lin();
      int colunas = camada.saida.col();

      for(i = 0; i < linhas; i++){
         for(j = 0; j < colunas; j++){
            grad = camada.gradSaida.dado(i, j);
            d = camada.saida.dado(i, j);
            d = 1 - (d * d);
            
            camada.derivada.editar(i, j, (d * grad));
         }
      }
   }

   @Override
   public void calcular(Convolucional camada){
      for(int i = 0; i < camada.somatorio.length; i++){
         super.aplicarFx(camada.somatorio[i], camada.saida[i]);
      }
   }

   @Override
   public void derivada(Convolucional camada){
      int i, j, k;
      double grad, d;
      int linhas, colunas;

      for(i = 0; i < camada.somatorio.length; i++){
         linhas = camada.somatorio[i].lin();
         colunas = camada.somatorio[i].col();
         for(j = 0; j < linhas; j++){
            for(k = 0; k < colunas; k++){
               grad = camada.gradSaida[i].dado(j, k);
               d = camada.saida[i].dado(j, k);
               d = 1 - (d * d);

               camada.derivada[i].editar(j, k, (grad * d));
            }
         }
      }
   }

   private double tanh(double x){
      return 2 / (1 + Math.exp(-2 * x)) - 1;
   }
}
