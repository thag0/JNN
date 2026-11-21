package jnn.core;

import jnn.inicializadores.Aleatorio;
import jnn.inicializadores.AleatorioPositivo;
import jnn.inicializadores.Constante;
import jnn.inicializadores.Gaussiano;
import jnn.inicializadores.GlorotNormal;
import jnn.inicializadores.GlorotUniforme;
import jnn.inicializadores.He;
import jnn.inicializadores.Identidade;
import jnn.inicializadores.Inicializador;
import jnn.inicializadores.LeCun;
import jnn.inicializadores.Zeros;
import jnn.metrics.loss.CossEntropy;
import jnn.metrics.loss.BinaryCrossEntropy;
import jnn.metrics.loss.MAE;
import jnn.metrics.loss.MSE;
import jnn.metrics.loss.MSLE;
import jnn.metrics.loss.Loss;
import jnn.metrics.loss.RMSE;
import jnn.otm.AdaGrad;
import jnn.otm.Adadelta;
import jnn.otm.Adam;
import jnn.otm.Lion;
import jnn.otm.Nadam;
import jnn.otm.Optimizer;
import jnn.otm.RMSProp;
import jnn.otm.SGD;

/**
 * Tradutor para as funções de ativação, otimizadores e funções de perda 
 * e inicializadores.
 */
public class JNNdict {

    /**
     * Tradutor para as funções de ativação, otimizadores, funções de perda 
     * e inicializadores.
     */
   	private JNNdict() {}

    /**
     * Remove caracteres especiais.
     * @param name nome de alguma instância que o dicionário lê.
     * @return nome tratado.
     */
	private static String tratarNome(String name) {
		return name.toLowerCase()
		.replace("_", "")
		.replace("-", "")
		.replace(".", "")
		.trim();
	}

    /**
     * Converte o otimizador lido em uma instância de 
     * otimizador correspondente.
     * @param optim tipo de de otimizador.
     * @return instância do otimizador lido.
     */
	public static Optimizer getOptim(Object optim) {
		JNNutils.validateNonNull(optim, "optim == null.");

		if (optim instanceof Optimizer) {
			return (Optimizer) optim;

		} else if (optim instanceof String) {
			String nome = (String) optim;
			switch (tratarNome(nome)){
				case "adadelta":  return new Adadelta();
				case "adagrad":   return new AdaGrad();
				case "adam":      return new Adam();
				case "lion":      return new Lion();
				case "nadam":     return new Nadam();
				case "rmsprop":   return new RMSProp();
				case "sgd":       return new SGD();
	
				default: throw new IllegalArgumentException(
					"\nOtimizador \"" + nome + "\" não encontado."
				);
			}

		} else {
			throw new IllegalArgumentException(
				"\nTipo de dado \"" + optim.getClass().getTypeName() + "\" não suportado."
			);
		}
	}

    /**
     * Converte a função de perda recebida em uma instância de função
     * de perda.
     * @param loss tipo de função de perda.
     * @return instância da função de perda lida.
     */
	public static Loss getLoss(Object loss) {
		JNNutils.validateNonNull(loss, "loss == null.");

		if (loss instanceof Loss) {
			return (Loss) loss;

		} else if (loss instanceof String) {
			String name = (String) loss;
			switch (tratarNome(name)) {
				case "mse"                : return new MSE();
				case "mae"                : return new MAE();
				case "msle"               : return new MSLE();
				case "rmse"               : return new RMSE();
				case "crossentropy"       : return new CossEntropy();
				case "binarycrossentropy" : return new BinaryCrossEntropy();
	
				default: throw new IllegalArgumentException(
					"\nFunção de perda \"" + name + "\" não encontada."
				);
			}           
		} else {
			throw new IllegalArgumentException(
				"\nTipo de dado \"" + loss.getClass().getTypeName() + "\" não suportado."
			);
		}
		
	}

    /**
     * Converte o inicializador recebido em uma instância de inicializador.
     * @param init tipo de inicializador.
     * @return instância do inicializador lido.
     */
	public static Inicializador getInitializer(Object init) {
		JNNutils.validateNonNull(init, "init == null.");

		if (init instanceof Inicializador) {
			return (Inicializador) init;
		
		} else if (init instanceof String) {
			String nome = (String) init;
			switch (tratarNome(nome)) {
				case "aleatorio"           : return new Aleatorio();
				case "aleatoriopositivo"   : return new AleatorioPositivo();
				case "constante"           : return new Constante();
				case "gaussiano"           : return new Gaussiano();
				case "glorotnormal"        : return new GlorotNormal();
				case "glorotuniforme"      : return new GlorotUniforme();
				case "he"                  : return new He();
				case "identidade"          : return new Identidade();
				case "lecun"               : return new LeCun();
				case "zeros"               : return new Zeros();
				
				default: throw new IllegalArgumentException(
					"\nInicializador \"" + nome + "\" não encontado."
				);
			}
		
		} else {
			throw new IllegalArgumentException(
				"\nTipo de dado \"" + init.getClass().getTypeName() + "\" não suportado."
			);
		}

	}
}
