package rna.ativacoes;

import rna.estrutura.Convolucional;
import rna.estrutura.Densa;

/**
 * Implementação da função de ativação ELU para uso dentro 
 * da {@code Rede Neural}.
 * <p>
 *    É possível configurar o valor de {@code alfa} para obter
 *    melhores resultados.
 * </p>
 */
public class ELU extends Ativacao{

   /**
    * Instancia a função de ativação ELU com 
    * seu valor de alfa configurável.
    * @param alfa novo valor alfa.
    */
   public ELU(double alfa){
      super.construir(
         (x) -> { return (x > 0) ? x : alfa * (Math.exp(x) - 1); },
         (x) -> { return (x > 0) ? 1 : alfa * Math.exp(x); }
      );
   }

   /**
    * Instancia a função de ativação ELU com 
    * seu valor de alfa padrão.
    * <p>
    *    O valor padrão para o alfa é {@code 0.01}.
    * </p>
    */
   public ELU(){
      this(0.01);
   }

   @Override
   public void calcular(Densa camada){
      super.aplicarFx(camada.somatorio, camada.saida);
   }

   @Override
   public void derivada(Densa camada){
      super.aplicarDx(camada.gradSaida, camada.somatorio, camada.derivada);
   }

   @Override
   public void calcular(Convolucional camada){
      for(int i = 0; i < camada.somatorio.length; i++){
         super.aplicarFx(camada.somatorio[i], camada.saida[i]);
      }
   }

   @Override
   public void derivada(Convolucional camada){
      for(int i = 0; i < camada.somatorio.length; i++){
         super.aplicarDx(camada.gradSaida[i], camada.somatorio[i], camada.derivada[i]);
      }
   }
}