package rna.camadas;

/**
 * <h2>
 *    Camada de entrada
 * </h2>
 * <p>
 *    A camada de entrada serve apenas para facilitar a criação de modelos 
 *    e não tem impacto após a compilação deles.
 * </p>
 * <p>
 *    Ela deve estar no início das camadas de um modelo para ser considerada,
 *    e no momento da compilação é destruída e servirá de base para a primeira 
 *    camada do modelo ser construída.
 * </p>
 * Exemplo
 * <pre>
 * Sequencial modelo = new Sequencial(Camada[]{
 *    Entrada(1, 28, 28),
 *    Flatten(),
 *    Densa(20, "sigmoid"),
 *    Densa(20, "sigmoid"),
 *    Densa(10, "softmax")
 * });
 * </pre>
 */
public class Entrada extends Camada{

   /**
    * Formato usado para entrada de um modelo.
    */
   int[] formato = {1, 1, 1, 1};

   /**
    * Inicializa um camada de entrada de acordo com o formato.
    * <p>
    *    A configuração do formato é dada por:
    * </p>
    * <pre>
    *    formato = (canais, profundidade, altura, largura)
    * </pre>
    * <p>
    *    Para formatos de tamanhos variados, serão considerados apenas 
    *    no elementos finais.
    * </p>
    *Exemplo:
    * <pre>
    *formato = (largura)
    *formato = (altura, largura)
    *formato = (profundidade, altura, largura)
    * </pre>
    * @param formato formato de entrada usado para o modelo em que a camada estiver.
    */
   public Entrada(int... formato){
      if(formato == null){
         throw new IllegalArgumentException(
            "\nFormato recebido é nulo."
         );
      }

      if(formato.length == 0){
         throw new UnsupportedOperationException(
            "\nO formato recebido deve conter ao menos um elemento."
         );
      }

      if(formato.length > 4){
         throw new UnsupportedOperationException(
            "\nO suporte dado ao formato de entrada é limitado (por enquanto) a" +
            " quantro elementos, recebido: " + formato.length
         );
      }

      int n1 = this.formato.length;
      int n2 = formato.length;
      for(int i = 0; i < n2; i++){
         this.formato[n1 - 1 - i] = formato[n2 - 1 - i];
      }
   }

   @Override
   public void construir(Object entrada){}

   @Override
   public void inicializar(){}

   @Override
   public void calcularSaida(Object entrada){}

   @Override
   public void calcularGradiente(Object gradSeguinte){}

   @Override
   public Object saida(){
      throw new UnsupportedOperationException("\nNão implementado.");
   }

   @Override
   public int[] formatoEntrada(){
      return this.formato;
   }

   @Override
   public int[] formatoSaida(){
      return this.formato;
   }

   @Override
   public int numParametros(){
      return 0;
   }
   
}