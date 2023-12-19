package rna.ativacoes;

import rna.estrutura.Convolucional;
import rna.estrutura.Densa;

public class Sigmoid extends Ativacao{

   public Sigmoid(){
      super.construir(this::sigmoid, null);
   }

   @Override
   public void calcular(Densa camada){
      super.aplicarFx(camada.somatorio, camada.saida);
   }

   @Override
   public void derivada(Densa camada){
      double grad, d;
      int i, j;
      int linhas = camada.saida.lin();
      int colunas = camada.saida.col();

      for(i = 0; i < linhas; i++){
         for(j = 0; j < colunas; j++){
            grad = camada.gradSaida.dado(i, j);
            d = camada.saida.dado(i, j);
            d = d * (1 - d);

            camada.derivada.editar(i, j, (grad * d));
         }
      }
   }

   @Override
   public void calcular(Convolucional camada){
      for(int i = 0; i < camada.somatorio.length; i++){
         aplicarFx(camada.somatorio[i], camada.saida[i]);
      }
   }

   @Override
   public void derivada(Convolucional camada){
      int i, j, k;
      double grad, d;

      for(i = 0; i < camada.somatorio.length; i++){
         for(j = 0; j < camada.somatorio[i].lin(); j++){
            for(k = 0; k < camada.somatorio[i].col(); k++){
               grad = camada.gradSaida[i].dado(j, k);
               d = camada.saida[i].dado(j, k);
               d = d * (1 - d);

               camada.derivada[i].editar(j, k, (grad * d));
            }
         }
      }
   }

   private double sigmoid(double x){
      return 1 / (1 + Math.exp(-x));
   }
}
