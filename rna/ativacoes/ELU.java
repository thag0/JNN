package rna.ativacoes;

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
    * Valor alfa da função ELU.
    */
   private double alfa;

   /**
    * Instancia a função de ativação ELU com 
    * seu valor de alfa configurável.
    * @param alfa novo valor alfa.
    */
   public ELU(double alfa){
      this.alfa = alfa;
      super.construir(this::elu, this::elud);
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
      super.aplicarFuncao(camada.somatorio, camada.saida);
   }

   @Override
   public void derivada(Densa camada){
      super.aplicarDerivada(camada.gradSaida, camada.somatorio, camada.derivada);
   }

   private double elu(double x){
      return x > 0 ? x : alfa * (Math.exp(x) - 1);
   }

   private double elud(double x){
      return x > 0 ? 1 : alfa * Math.exp(x);
   }
}
