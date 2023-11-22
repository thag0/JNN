package rna.otimizadores;

import rna.core.Matriz;
import rna.estrutura.CamadaDensa;

/**
 * Classe que implementa o algoritmo de Descida do Gradiente para otimização de redes neurais.
 * Atualiza diretamente os pesos da rede com base no gradiente.
 */
public class GD extends Otimizador{

   Matriz mat = new Matriz();

   /**
    * Valor de taxa de aprendizagem do otimizador.
    */
   private double taxaAprendizagem;

   /**
    * Inicializa uma nova instância de otimizador da <strong> Descida do Gradiente </strong>
    * usando os valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
    */
   public GD(double tA){
      this.taxaAprendizagem = tA;
   }

   /**
    * Inicializa uma nova instância de otimizador da <strong> Descida do Gradiente </strong>.
    * <p>
    *    Os hiperparâmetros do GD serão inicializados com os valores padrão, que são:
    * </p>
    * {@code taxaAprendizagem = 0.01}
    */
   public GD(){
      this(0.01);
   }

   @Override
   public void inicializar(CamadaDensa[] redec){
      
   }

   /**
    * Aplica o algoritmo do Gradiente descendente para cada peso da rede neural.
    * <p>
    *    O Gradiente descendente funciona usando a seguinte expressão:
    * </p>
    * <pre>
    *    p[i] -= g[i] * tA
    * </pre>
    * Onde:
    * <p>
    *    {@code p} - peso que será atualizado.
    * </p>
    *    {@code g} - gradiente correspondente a conexão do peso que será
    *    atualizado.
    * </p>
    * <p>
    *    {@code tA} - taxa de aprendizagem.
    * </p>
    */
    @Override
   public void atualizar(CamadaDensa[] redec){
      for(CamadaDensa camada : redec){
         mat.escalar(camada.gradientes, taxaAprendizagem, camada.gradientes);
         mat.escalar(camada.gradientes, -1, camada.gradientes);
         mat.sub(camada.pesos, camada.gradientes, camada.pesos);

         if(camada.temBias()){
            mat.escalar(camada.erros, taxaAprendizagem, camada.erros);
            mat.escalar(camada.erros, -1, camada.erros);
            mat.sub(camada.bias, camada.erros, camada.bias);
         }
      } 
   }

   @Override
   public String info(){
      String buffer = "";

      String espacamento = "    ";
      buffer += espacamento + "TaxaAprendizagem: " + this.taxaAprendizagem + "\n";

      return buffer;
   }
   
}
