package rna.ativacoes;

import rna.estrutura.CamadaDensa;

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
   public void calcular(CamadaDensa camada){
      for(int i = 0; i < camada.saida.length; i++){
         for(int j = 0; j < camada.saida[i].length; j++){
            camada.saida[i][j] = elu(camada.somatorio[i][j]);
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      for(int i = 0; i < camada.derivada.length; i++){
         for(int j = 0; j < camada.derivada[i].length; j++){
            camada.derivada[i][j] = derivada(camada.somatorio[i][j]);
         }
      }
   }

   private double elu(double x){
      return x > 0 ? x : alfa * Math.exp(x) - 1;
   }

   private double derivada(double x){
      return x > 0 ? 1 : alfa * Math.exp(x);
   }
}
