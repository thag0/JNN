package rna.otimizadores;

import rna.estrutura.CamadaDensa;

/**
 * Implementação do algoritmo de otimização AMSGrad, que é uma variação do 
 * algoritmo Adam que resolve um problema de convergência em potencial do Adam.
 * <p>
 * 	Os hiperparâmetros do AMSGrad podem ser ajustados para controlar o 
 * 	comportamento do otimizador durante o treinamento.
 * </p>
 */
public class AMSGrad extends Otimizador{

   /**
    * Valor de taxa de aprendizagem do otimizador.
    */
	private double taxaAprendizagem;

	/**
	 * Usado para evitar divisão por zero.
	 */
	private double epsilon;

	/**
	 * Decaimento do momentum de primeira ordem.
	 */
	private double beta1;

	/**
	 * Decaimento do momentum de segunda ordem.
	 */
	private double beta2;

   /**
    * Coeficientes de momentum.
    */
	private double[][][] m;

   /**
    * Coeficientes de momentum para os bias.
    */
	private double[][][] mb;

	/**
	 * Coeficientes de momentum de segunda orgem.
	 */
	private double[][][] v;

	/**
	 * Coeficientes de momentum de segunda orgem para os bias.
	 */
	private double[][][] vb;

	/**
	 * Coeficientes de momentum de segunda orgem corrigidos.
	 */
	private double[][][] vc;

	/**
	 * Coeficientes de momentum de segunda orgem corrigidos para os bias.
	 */
	private double[][][] vcb;

	/**
	 * Contador de iterações.
	 */
	private long interacoes;

	/**
	 * Inicializa uma nova instância de otimizador <strong> AMSGrad </strong> usando os valores de
	 * hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
	 * @param epsilon usado para evitar a divisão por zero.
	 * @param beta1 decaimento do momento.
	 * @param beta2 decaimento do momento de segunda ordem.
	 */
	public AMSGrad(double tA, double beta1, double beta2, double epsilon){
		this.taxaAprendizagem = tA;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.epsilon = epsilon;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> AMSGrad </strong>.
	 * <p>
	 * Os hiperparâmetros do AMSGrad serão inicializados com os valores padrão, que
	 * são:
    * <p>
    *    {@code taxaAprendizagem = 0.01}
    * </p>
    * <p>
    *    {@code beta1 = 0.9}
    * </p>
    * <p>
    *    {@code beta2 = 0.999}
    * </p>
    * <p>
    *    {@code epsilon = 1e-7}
    * </p>
	 */
	public AMSGrad(){
		this(0.01, 0.9, 0.999, 1e-7);
	}

	@Override
   public void inicializar(CamadaDensa[] redec){
      this.m = new double[redec.length][][];
      this.v = new double[redec.length][][];
      this.vc = new double[redec.length][][];
      
		this.mb = new double[redec.length][][];
      this.vb = new double[redec.length][][];
      this.vcb = new double[redec.length][][];
   
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         this.m[i] = new double[camada.pesos.length][camada.pesos[0].length];
         this.v[i] = new double[camada.pesos.length][camada.pesos[0].length];
         this.vc[i] = new double[camada.pesos.length][camada.pesos[0].length];
         
         if(camada.temBias()){
            this.mb[i] = new double[camada.bias.length][camada.bias[0].length];
            this.vb[i] = new double[camada.bias.length][camada.bias[0].length];
            this.vcb[i] = new double[camada.bias.length][camada.bias[0].length];
         }
      }
   }

   /**
    * Aplica o algoritmo do AMSGrad para cada peso da rede neural.
    * <p>
    *    O AMSGrad funciona usando a seguinte expressão:
    * </p>
    * <pre>
    *    p[i] -= (tA * mc) / ((√ vc) + eps)
    * </pre>
    * Onde:
    * <p>
    *    {@code p} - peso que será atualizado.
    * </p>
    * <p>
    *    {@code tA} - valor de taxa de aprendizagem (learning rate).
    * </p>
    * <p>
    *    {@code mc} - valor de momentum corrigido.
    * </p>
    * <p>
    *    {@code vc} - valor de momentum de segunda ordem corrigido.
    * </p>
    * Os valores de momentum corrigido (mc) e momentum de segunda ordem
    * corrigido (vc) se dão por:
    * <pre>
    *    mc = m[i] / (1 - beta1ⁱ)
    * </pre>
    * <pre>
    *    vc = vC[i] / (1 - beta2ⁱ)
    * </pre>
    * Onde:
    * <p>
    *    {@code m} - valor de momentum correspondete a conexão do peso que está
    *     sendo atualizado.
    * </p>
	 *	<p>
	 *		{@code max2ordem} - valor máximo de segunda ordem calculado.
	 *	</p>
    * <p>
    *    {@code vC} - valor de momentum de segunda ordem corrigido correspondente a 
	 *		conexão do peso que está sendo atualizado.
    * </p>
    * <p>
    *    {@code i} - contador de interações do otimizador.
    * </p>
	 * O valor de momentum de segunda ordem corrigido (vC) é dado por:
	 * <pre>
	 * vC[i] = max(vC[i], v[i])
	 * </pre>
	 * Onde:
	 * <p>
    *    {@code v} - coeficiente de momentum de segunda ordem correspondente a
	 *		conexão do peso que está sendo atualizado.
    * </p>
    */
	@Override
	public void atualizar(CamadaDensa[] redec){
		interacoes++;
		
		double g;
		double forcaB1 = (1 - Math.pow(beta1, interacoes));
		double forcaB2 = (1 - Math.pow(beta2, interacoes));

		for(int i = 0; i < redec.length; i++){
			CamadaDensa camada = redec[i];

			for(int j = 0; j < camada.pesos.length; j++){
				for(int k = 0; k < camada.pesos[j].length; k++){
					g = camada.gradientes[j][k];

					m[i][j][k] = (beta1 * m[i][j][k]) + ((1 - beta1) * g);
					v[i][j][k] = (beta2 * v[i][j][k]) + ((1 - beta2) * g * g);
					vc[i][j][k] = Math.max(vc[i][j][k], v[i][j][k]);

					camada.pesos[j][k] += calcular(m[i][j][k], vc[i][j][k], forcaB1, forcaB2);
				}			
			}

         if(camada.temBias()){
            for(int j = 0; j < camada.bias.length; j++){
               for(int k = 0; k < camada.bias[j].length; k++){
                  g = camada.erros[j][k];

						mb[i][j][k] = (beta1 * mb[i][j][k]) + ((1 - beta1) * g);
						vb[i][j][k] = (beta2 * vb[i][j][k]) + ((1 - beta2) * g * g);
						vcb[i][j][k] = Math.max(vcb[i][j][k], vb[i][j][k]);

                  camada.bias[j][k] += calcular(mb[i][j][k], vcb[i][j][k], forcaB1, forcaB2);
               }
            }
         } 
		}
  	}

	  private double calcular(double m, double vc, double forcaB1, double forcaB2){
		double mChapeu = m / forcaB1;
		double vChapeu = vc / forcaB2;
		return (taxaAprendizagem * mChapeu) / (Math.sqrt(vChapeu) + epsilon);
   }

	@Override
	public String info(){
		String buffer = "";

		String espacamento = "    ";
		buffer += espacamento + "TaxaAprendizagem: " + this.taxaAprendizagem + "\n";
		buffer += espacamento + "Beta1: " + this.beta1 + "\n";
		buffer += espacamento + "Beta2: " + this.beta2 + "\n";
		buffer += espacamento + "Epsilon: " + this.epsilon + "\n";

		return buffer;
	}

}
