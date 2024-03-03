package rna.otimizadores;

import rna.camadas.Camada;

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
 *    v -= (tA * mc) / ((√ vc) + eps)
 * </pre>
 * Onde:
 * <p>
 *    {@code p} - variável que será otimizada.
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
 *    mc = m / (1 - beta1ⁱ)
 * </pre>
 * <pre>
 *    vc = vC / (1 - beta2ⁱ)
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
 * vC = max(vC, v)
 * </pre>
 * Onde:
 * <p>
 *    {@code v} - coeficiente de momentum de segunda ordem correspondente a
 *		conexão do peso que está sendo atualizado.
 * </p>
 */
public class AMSGrad extends Otimizador{

   /**
    * Valor padrão para a taxa de aprendizagem do otimizador.
    */
	private static final double PADRAO_TA = 0.001;

	/**
	 * Valor padrão para o decaimento do momento de primeira ordem.
	 */
	private static final double PADRAO_BETA1 = 0.95;
 
	/**
	 * Valor padrão para o decaimento do momento de segunda ordem.
	 */
	private static final double PADRAO_BETA2 = 0.999;
	 
	/**
	 * Valor padrão para epsilon.
	 */
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
	 * @param beta1 decaimento do momento.
	 * @param beta2 decaimento do momento de segunda ordem.
	 * @param epsilon usado para evitar a divisão por zero.
	 */
	public AMSGrad(double tA, double beta1, double beta2, double epsilon){
      if(tA <= 0){
         throw new IllegalArgumentException(
            "\nTaxa de aprendizagem (" + tA + "), inválida."
         );
      }
      if(beta1 <= 0){
         throw new IllegalArgumentException(
            "\nTaxa de decaimento de primeira ordem (" + beta1 + "), inválida."
         );
      }
      if(beta2 <= 0){
         throw new IllegalArgumentException(
            "\nTaxa de decaimento de segunda ordem (" + beta2 + "), inválida."
         );
      }
      if(epsilon <= 0){
         throw new IllegalArgumentException(
            "\nEpsilon (" + epsilon + "), inválido."
         );
      }
		
		this.taxaAprendizagem = tA;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.epsilon = epsilon;
	}

	/**
	 * Inicializa uma nova instância de otimizador <strong> AMSGrad </strong> usando os 
	 * valores de hiperparâmetros fornecidos.
    * @param tA valor de taxa de aprendizagem.
	 * @param beta1 decaimento do momento.
	 * @param beta2 decaimento do momento de segunda ordem.
	 */
	public AMSGrad(double tA, double beta1, double beta2){
      this(tA, beta1, beta2, PADRAO_EPS);
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
   public void construir(Camada[] camadas){
      int nKernel = 0;
      int nBias = 0;
      
      for(Camada camada : camadas){
			if(camada.treinavel == false) continue;

         nKernel += camada.kernelParaArray().length;
         if(camada.temBias()){
            nBias += camada.biasParaArray().length;
         }         
      }

      this.m  = new double[nKernel];
      this.v  = new double[nKernel];
      this.vc  = new double[nKernel];
      this.mb = new double[nBias];
      this.vb = new double[nBias];
      this.vcb = new double[nBias];
		this.construido = true;//otimizador pode ser usado
   }

	@Override
	public void atualizar(Camada[] camadas){
		verificarConstrucao();
		
		int idKernel = 0, idBias = 0;

		interacoes++;
		double forcaB1 = (1 - Math.pow(beta1, interacoes));
		double forcaB2 = (1 - Math.pow(beta2, interacoes));
		
		for(Camada camada : camadas){
			if(camada.treinavel == false) continue;

			double[] kernel = camada.kernelParaArray();
			double[] gradK = camada.gradKernelParaArray();
			idKernel = calcular(kernel, gradK, m, v, vc, forcaB1, forcaB2, idKernel);
			camada.editarKernel(kernel);

         if(camada.temBias()){
				double[] bias = camada.biasParaArray();
				double[] gradB = camada.gradBias();
				idBias = calcular(bias, gradB, mb, vb, vcb, forcaB1, forcaB2, idBias);
				camada.editarBias(bias);
         } 
		}
  	}

   /**
    * Atualiza as variáveis usando o gradiente pré calculado.
    * @param vars variáveis que serão atualizadas.
    * @param grads gradientes das variáveis.
    * @param m coeficientes de momentum de primeira ordem das variáveis.
    * @param v coeficientes de momentum de segunda ordem das variáveis.
	 * @param vc coeficientes de momentum de segunda ordem corrigidos.
    * @param forcaB1 força do decaimento do momentum de primeira ordem.
    * @param forcaB2 força do decaimento do momentum de segunda ordem.
    * @param id índice inicial das variáveis dentro do array de momentums.
    * @return índice final após as atualizações.
	 */
	private int calcular(double[] vars, double[] grads, double[] m, double[] v, double[] vc, double forcaB1, double forcaB2, int id){
		double mChapeu, vChapeu, g;

		for(int i = 0; i < vars.length; i++){
			g = grads[i];
			m[id] = (beta1 * m[id]) + ((1 - beta1) * g);
			v[id] = (beta2 * v[id]) + ((1 - beta2) * (g*g));
			vc[id] = Math.max(vc[id], v[id]);

			mChapeu = m[id] / forcaB1;
			vChapeu = v[id] / forcaB2;
			vars[i] -= (mChapeu * taxaAprendizagem) / (Math.sqrt(vChapeu) + epsilon);

			id++;
		}

		return id;
	}

	@Override
	public String info(){
		super.verificarConstrucao();
      super.construirInfo();
      
      super.addInfo("TaxaAprendizagem: " + this.taxaAprendizagem);
      super.addInfo("Beta1: " + this.beta1);
      super.addInfo("Beta2: " + this.beta2);
      super.addInfo("Epsilon: " + this.epsilon);

      return super.info();
	}

}
