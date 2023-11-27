package rna.ativacoes;

import rna.estrutura.CamadaDensa;

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
   public void calcular(CamadaDensa camada){
      int i, j;
      double s;

      for(i = 0; i < camada.saida.lin; i++){
         for(j = 0; j < camada.saida.col; j++){
            s = leakyRelu(camada.somatorio.dado(i, j));
            camada.saida.editar(i, j, s);
         }
      }
   }

   @Override
   public void derivada(CamadaDensa camada){
      int i, j;
      double d;

      for(i = 0; i < camada.derivada.lin; i++){
         for(j = 0; j < camada.derivada.col; j++){
            d = derivada(camada.somatorio.dado(i, j));
            camada.derivada.editar(i, j, d);
         }
      }
   }

   private double leakyRelu(double x){
      return x > 0 ? x : alfa * x;
   }

   private double derivada(double x){
      return x > 0 ? 1 : alfa;
   }
}
