package jnn.core;

import jnn.acts.Argmax;
import jnn.acts.Atan;
import jnn.acts.Ativacao;
import jnn.acts.ELU;
import jnn.acts.GELU;
import jnn.acts.LeakyReLU;
import jnn.acts.Linear;
import jnn.acts.ReLU;
import jnn.acts.SELU;
import jnn.acts.Seno;
import jnn.acts.Sigmoid;
import jnn.acts.SoftPlus;
import jnn.acts.Softmax;
import jnn.acts.Swish;
import jnn.acts.TanH;
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
import jnn.metrica.perda.EntropiaCruzada;
import jnn.metrica.perda.EntropiaCruzadaBinaria;
import jnn.metrica.perda.MAE;
import jnn.metrica.perda.MSE;
import jnn.metrica.perda.MSLE;
import jnn.metrica.perda.Perda;
import jnn.metrica.perda.RMSE;
import jnn.otm.AdaGrad;
import jnn.otm.Adadelta;
import jnn.otm.Adam;
import jnn.otm.Lion;
import jnn.otm.Nadam;
import jnn.otm.Otimizador;
import jnn.otm.RMSProp;
import jnn.otm.SGD;

/**
 * Tradutor para as funções de ativação, otimizadores e funções de perda 
 * e inicializadores.
 */
public class Dicionario {

    /**
     * Tradutor para as funções de ativação, otimizadores, funções de perda 
     * e inicializadores.
     */
   	public Dicionario() {}

    /**
     * Remove caracteres especiais.
     * @param nome nome de alguma instância que o dicionário lê.
     * @return nome tratado.
     */
	private String tratarNome(String nome) {
		nome = nome.trim();
		nome = nome.replace("_", "");
		nome = nome.replace("-", "");
		nome = nome.replace(".", "");

		return nome.toLowerCase();
	}

    /**
     * Converte a ativação lida em uma instância de função
     * de ativação correspondente.
     * @param act tipo função de ativação.
     * @return instância da função de ativação lida.
     */
	public Ativacao getAtivacao(Object act) {
		JNNutils.validarNaoNulo(act, "act == null.");
		
		if (act instanceof Ativacao) {
			return (Ativacao) act;
		
		} else if (act instanceof String) {
			String nome = (String) act;
			switch (tratarNome(nome)) {
				case "argmax"     : return new Argmax();
				case "elu"        : return new ELU();
				case "gelu"       : return new GELU();
				case "leakyrelu"  : return new LeakyReLU();
				case "linear"     : return new Linear();
				case "relu"       : return new ReLU();
				case "seno"       : return new Seno();
				case "sigmoid"    : return new Sigmoid();
				case "softmax"    : return new Softmax();
				case "softplus"   : return new SoftPlus();
				case "swish"      : return new Swish();
				case "tanh"       : return new TanH();
				case "atan"       : return new Atan();
				case "selu"       : return new SELU();
	
				default: throw new IllegalArgumentException(
					"\nAtivação \"" + nome + "\" não encontada."
				);
			}

		} else {
			throw new IllegalArgumentException(
				"\nTipo de dado \"" + act.getClass().getTypeName() + "\" não suportado."
			);
		}
	}

    /**
     * Converte o otimizador lido em uma instância de 
     * otimizador correspondente.
     * @param otm tipode de otimizador.
     * @return instância do otimizador lido.
     */
	public Otimizador getOtimizador(Object otm) {
		JNNutils.validarNaoNulo(otm, "otm == null.");

		if (otm instanceof Otimizador) {
			return (Otimizador) otm;

		} else if (otm instanceof String) {
			String nome = (String) otm;
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
				"\nTipo de dado \"" + otm.getClass().getTypeName() + "\" não suportado."
			);
		}
	}

    /**
     * Converte a função de perda recebida em uma instância de função
     * de perda.
     * @param loss tipo de função de perda.
     * @return instância da função de perda lida.
     */
	public Perda getPerda(Object loss) {
		JNNutils.validarNaoNulo(loss, "loss == null.");

		if (loss instanceof Perda) {
			return (Perda) loss;

		} else if (loss instanceof String) {
			String nome = (String) loss;
			switch (tratarNome(nome)) {
				case "mse"                    : return new MSE();
				case "mae"                    : return new MAE();
				case "msle"                   : return new MSLE();
				case "rmse"                   : return new RMSE();
				case "entropiacruzada"        : return new EntropiaCruzada();
				case "entropiacruzadabinaria" : return new EntropiaCruzadaBinaria();
	
				default: throw new IllegalArgumentException(
					"\nFunção de perda \"" + nome + "\" não encontada."
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
     * @param ini tipo de inicializador.
     * @return instância do inicializador lido.
     */
	public Inicializador getInicializador(Object ini) {
		JNNutils.validarNaoNulo(ini, "ini == null.");

		if (ini instanceof Inicializador) {
			return (Inicializador) ini;
		
		} else if (ini instanceof String) {
			String nome = (String) ini;
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
				"\nTipo de dado \"" + ini.getClass().getTypeName() + "\" não suportado."
			);
		}

	}
}
