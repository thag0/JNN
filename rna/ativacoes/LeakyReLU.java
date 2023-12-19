package rna.ativacoes;

import rna.estrutura.Convolucional;
import rna.estrutura.Densa;

/**
 * Implementação da função de ativação LeakyReLU para uso dentro 
 * da {@code Rede Neural}.
 * <p>
 *    É possível configurar o valor de {@code alfa} para obter
 *    melhores resultados.
 * </p>
 */
public class LeakyReLU extends Ativacao{

   /**
    * Valor alfa da função LeakyReLU.
    */
   private double alfa;

   /**
    * Instancia a função de ativação LeakyReLU com seu valor de alfa configurável.
    * <p>
    *    A ativação LeakyReLU funciona semelhante a função ReLU, retornando o próprio 
    *    valor recebido caso ele seja maior que um, mas caso contrário ela retorna um 
    *    pequeno valor alfa que será multiplicado pela saída.
    * </p>
    * @param alfa novo valor alfa.
    */
   public LeakyReLU(double alfa){
      this.alfa = alfa;
      super.construir(this::leakyRelu, this::leakyRelud);
   }

   /**
    * Instancia a função de ativação LeakyReLU com o valor de alfa padrão.
    * <p>
    *    A ativação LeakyReLU funciona semelhante a função ReLU, retornando o próprio 
    *    valor recebido caso ele seja maior que um, mas caso contrário ela retorna um 
    *    pequeno valor alfa que será multiplicado pela saída.
    * </p>
    * <p>
    *    O valor padrão para o alfa é {@code 0.01}.
    * </p>
    */
   public LeakyReLU(){
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
         aplicarFx(camada.somatorio[i], camada.saida[i]);
      }
   }

   @Override
   public void derivada(Convolucional camada){
      for(int i = 0; i < camada.somatorio.length; i++){
         aplicarDx(camada.gradSaida[i], camada.somatorio[i], camada.derivada[i]);
      }
   }

   private double leakyRelu(double x){
      return x > 0 ? x : alfa * x;
   }

   private double leakyRelud(double x){
      return x > 0 ? 1 : alfa;
   }
}
