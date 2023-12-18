package rna.otimizadores;

import rna.estrutura.Camada;

/**
 * Implementação do algoritmo de otimização AMSGrad, que é uma variação do 
 * algoritmo Adam que resolve um problema de convergência em potencial do Adam.
 * <p>
 * 	Os hiperparâmetros do AMSGrad podem ser ajustados para controlar o 
 * 	comportamento do otimizador durante o treinamento.
 * </p>
 * <p>
 *    O AMSGrad funciona usando a seguinte expressão:
 * </p>
 * <pre>
 *    v[i][j] -= (tA * mc) / ((√ vc) + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code p} - variável que será otimizada (kernel, bias).
 * </p>
 * <p>
 *    {@code tA} - valor de taxa de aprendizagem.
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
 *    mc = m[i][j] / (1 - beta1ⁱ)
 * </pre>
 * <pre>
 *    vc = vC[i][j] / (1 - beta2ⁱ)
 * </pre>
 * Onde:
 * <p>
 *    {@code m} - valor de momentum correspondete a variável que será otimizada.
 * </p>
 * <p>
 *    {@code vC} - valor de momentum de segunda ordem corrigido correspondente 
 * 	a variável que será otimizada.
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
public class AMSGrad extends Otimizador{
	private static final double PADRAO_TA = 0.001;
   private static final double PADRAO_BETA1 = 0.95;
   private static final double PADRAO_BETA2 = 0.999;
   private static final double PADRAO_EPS = 1e-7;

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
    * Coeficientes de momentum para os kernels.
    */
	private double[] m;

   /**
    * Coeficientes de momentum para os bias.
    */
	private double[] mb;

	/**
	 * Coeficientes de momentum de segunda orgem para os kernels.
	 */
	private double[] v;

	/**
	 * Coeficientes de momentum de segunda orgem para os bias.
	 */
	private double[] vb;

	/**
	 * Coeficientes de momentum de segunda orgem corrigidos para os kernels.
	 */
	private double[] vc;

	/**
	 * Coeficientes de momentum de segunda orgem corrigidos para os bias.
	 */
	private double[] vcb;

	/**
	 * Contador de iterações.
	 */
	private long interacoes;

	/**
	 * Inicializa uma nova instância de otimizador <strong> AMSGrad </strong> usando os 
	 * valores de hiperparâmetros fornecidos.
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
	 * Inicializa uma nova instância de otimizador <strong> AMSGrad </strong> usando os 
	 * valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
	 */
	public AMSGrad(double tA){
		this(tA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> AMSGrad </strong>.
	 * <p>
	 * 	Os hiperparâmetros do AMSGrad serão inicializados com os valores padrão.
	 * </p>
	 */
	public AMSGrad(){
		this(PADRAO_TA, PADRAO_BETA1, PADRAO_BETA2, PADRAO_EPS);
	}

	@Override
   public void inicializar(Camada[] redec){
      int nKernel = 0;
      int nBias = 0;
      
      for(Camada camada : redec){
			if(camada.treinavel == false) continue;

         nKernel += camada.obterKernel().length;
         if(camada.temBias()){
            nBias += camada.obterBias().length;
         }         
      }

      this.m  = new double[nKernel];
      this.v  = new double[nKernel];
      this.vc  = new double[nKernel];
      this.mb = new double[nBias];
      this.vb = new double[nBias];
      this.vcb = new double[nBias];
   }

	@Override
	public void atualizar(Camada[] redec){
		int idKernel = 0, idBias = 0;
		double g, mChapeu, vChapeu;

		interacoes++;
		double forcaB1 = (1 - Math.pow(beta1, interacoes));
		double forcaB2 = (1 - Math.pow(beta2, interacoes));
		
		for(Camada camada : redec){
			if(camada.treinavel == false) continue;

			double[] kernel = camada.obterKernel();
			double[] gradK = camada.obterGradKernel();

			for(int i = 0; i < kernel.length; i++){
				g = gradK[i];
				m[idKernel] = (beta1 * m[idKernel]) + ((1 - beta1) * g);
				v[idKernel] = (beta2 * v[idKernel]) + ((1 - beta2) * (g*g));
				vc[idKernel] = Math.max(vc[idKernel], v[idKernel]);

				mChapeu = m[idKernel] / forcaB1;
				vChapeu = v[idKernel] / forcaB2;
				kernel[i] += (mChapeu * taxaAprendizagem) / (Math.sqrt(vChapeu) + epsilon);
				idKernel++;
			}
			camada.editarKernel(kernel);

         if(camada.temBias()){
				double[] bias = camada.obterBias();
				double[] gradB = camada.obterGradBias();

				for(int i = 0; i < bias.length; i++){
					g = gradB[i];
					mb[idBias] = (beta1 * mb[idBias]) + ((1 - beta1) * g);
					vb[idBias] = (beta2 * vb[idBias]) + ((1 - beta2) * (g*g));
					vcb[idBias] = Math.max(vcb[idBias], vb[idBias]);

					mChapeu = mb[idBias] / forcaB1;
					vChapeu = vb[idBias] / forcaB2;
					bias[i] += (mChapeu * taxaAprendizagem) / (Math.sqrt(vChapeu) + epsilon);
					idBias++;	
				}
				camada.editarBias(bias);
         } 
		}
  	}

	@Override
	public String info(){
      String espacamento = "    ";
      
      String buffer = "";
      buffer += espacamento + "Otimizador: " +  this.nome() + "\n";
		buffer += espacamento + "TaxaAprendizagem: " + this.taxaAprendizagem + "\n";
		buffer += espacamento + "Beta1: " + this.beta1 + "\n";
		buffer += espacamento + "Beta2: " + this.beta2 + "\n";
		buffer += espacamento + "Epsilon: " + this.epsilon + "\n";

		return buffer;
	}

}
