package rna.otimizadores;

import rna.core.Mat;
import rna.estrutura.CamadaDensa;

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
 *    {@code p} - variável que será otimizada (peso ou bias).
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
	private static final double PADRAO_TA = 0.01;
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
    * Coeficientes de momentum.
    */
	private Mat[] m;

   /**
    * Coeficientes de momentum para os bias.
    */
	private Mat[] mb;

	/**
	 * Coeficientes de momentum de segunda orgem.
	 */
	private Mat[] v;

	/**
	 * Coeficientes de momentum de segunda orgem para os bias.
	 */
	private Mat[] vb;

	/**
	 * Coeficientes de momentum de segunda orgem corrigidos.
	 */
	private Mat[] vc;

	/**
	 * Coeficientes de momentum de segunda orgem corrigidos para os bias.
	 */
	private Mat[] vcb;

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
   public void inicializar(CamadaDensa[] redec){
      this.m  = new Mat[redec.length];
      this.v  = new Mat[redec.length];
      this.vc = new Mat[redec.length];
      
		this.mb  = new Mat[redec.length];
      this.vb  = new Mat[redec.length];
      this.vcb = new Mat[redec.length];
   
      for(int i = 0; i < redec.length; i++){
         CamadaDensa camada = redec[i];

         this.m[i]  = new Mat(camada.pesos.lin, camada.pesos.col);
         this.v[i]  = new Mat(camada.pesos.lin, camada.pesos.col);
         this.vc[i] = new Mat(camada.pesos.lin, camada.pesos.col);
         
         if(camada.temBias()){
            this.mb[i]  = new Mat(camada.bias.lin, camada.bias.col);
            this.vb[i]  = new Mat(camada.bias.lin, camada.bias.col);
            this.vcb[i] = new Mat(camada.bias.lin, camada.bias.col);
         }
      }
   }

	@Override
	public void atualizar(CamadaDensa[] redec){
		interacoes++;
		double forcaB1 = (1 - Math.pow(beta1, interacoes));
		double forcaB2 = (1 - Math.pow(beta2, interacoes));
		
		for(int i = 0; i < redec.length; i++){
			CamadaDensa camada = redec[i];
			Mat pesos = camada.pesos;
			Mat grads = camada.gradPesos;

			for(int j = 0; j < pesos.lin; j++){
				for(int k = 0; k < pesos.col; k++){
					calcular(pesos, grads, m[i], v[i], vc[i], j, k, forcaB1, forcaB2);
				}			
			}

         if(camada.temBias()){
				Mat bias = camada.bias;
				Mat gradsB = camada.gradienteSaida;

            for(int j = 0; j < bias.lin; j++){
               for(int k = 0; k < bias.col; k++){
                  calcular(bias, gradsB, mb[i], vb[i], vcb[i], j, k, forcaB1, forcaB2);
               }
            }
         } 
		}
  	}

	private void calcular(Mat var, Mat grad, Mat m, Mat v, Mat vc, int lin, int col, double fb1, double fb2){
		double g = grad.dado(lin, col);

		double m2 = (beta1 * m.dado(lin, col)) + ((1 - beta1) * g);
		double v2 = (beta2 * v.dado(lin, col)) + ((1 - beta2) * (g*g));
		m.editar(lin, col, m2);
		v.editar(lin, col, v2);
		vc.editar(lin, col, Math.max(v.dado(lin, col), v2));

		double mChapeu = m.dado(lin, col) / fb1;
		double vChapeu = vc.dado(lin, col) / fb2;
		double att = (mChapeu * taxaAprendizagem) / (Math.sqrt(vChapeu) + epsilon);
		var.add(lin, col, att);
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
