package jnn.core;

import jnn.ativacoes.Argmax;
import jnn.ativacoes.Atan;
import jnn.ativacoes.Ativacao;
import jnn.ativacoes.ELU;
import jnn.ativacoes.GELU;
import jnn.ativacoes.LeakyReLU;
import jnn.ativacoes.Linear;
import jnn.ativacoes.ReLU;
import jnn.ativacoes.SELU;
import jnn.ativacoes.Seno;
import jnn.ativacoes.Sigmoid;
import jnn.ativacoes.SoftPlus;
import jnn.ativacoes.Softmax;
import jnn.ativacoes.Swish;
import jnn.ativacoes.TanH;
import jnn.avaliacao.perda.EntropiaCruzada;
import jnn.avaliacao.perda.EntropiaCruzadaBinaria;
import jnn.avaliacao.perda.MAE;
import jnn.avaliacao.perda.MSE;
import jnn.avaliacao.perda.MSLE;
import jnn.avaliacao.perda.Perda;
import jnn.avaliacao.perda.RMSE;
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
import jnn.otimizadores.AdaGrad;
import jnn.otimizadores.Adadelta;
import jnn.otimizadores.Adam;
import jnn.otimizadores.GD;
import jnn.otimizadores.Lion;
import jnn.otimizadores.Nadam;
import jnn.otimizadores.Otimizador;
import jnn.otimizadores.RMSProp;
import jnn.otimizadores.SGD;

/**
 * Tradutor para as funções de ativação, otimizadores e funções de perda 
 * e inicializadores.
 */
public class Dicionario {

	/**
	 * Utilitário.
	 */
	private Utils utils = new Utils();

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
     * @param ativacao tipo função de ativação.
     * @return instância da função de ativação lida.
     */
	public Ativacao getAtivacao(Object ativacao) {
		utils.validarNaoNulo(ativacao, "Ativação nula.");
		
		if (ativacao instanceof Ativacao) {
			return (Ativacao) ativacao;
		
		} else if (ativacao instanceof String) {
			String nome = (String) ativacao;
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
				"\nTipo de dado \"" + ativacao.getClass().getTypeName() + "\" não suportado."
			);
		}
	}

    /**
     * Converte o otimizador lido em uma instância de 
     * otimizador correspondente.
     * @param otimizador tipode de otimizador.
     * @return instância do otimizador lido.
     */
	public Otimizador getOtimizador(Object otimizador) {
		utils.validarNaoNulo(otimizador, "Otimizador nulo.");

		if (otimizador instanceof Otimizador) {
			return (Otimizador) otimizador;

		} else if (otimizador instanceof String) {
			String nome = (String) otimizador;
			switch (tratarNome(nome)){
				case "adadelta":  return new Adadelta();
				case "adagrad":   return new AdaGrad();
				case "adam":      return new Adam();
				case "gd":        return new GD();
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
				"\nTipo de dado \"" + otimizador.getClass().getTypeName() + "\" não suportado."
			);
		}
	}

    /**
     * Converte a função de perda recebida em uma instância de função
     * de perda.
     * @param perda tipo de função de perda.
     * @return instância da função de perda lida.
     */
	public Perda getPerda(Object perda) {
		utils.validarNaoNulo(perda, "Função de perda nula.");

		if (perda instanceof Perda) {
			return (Perda) perda;

		} else if (perda instanceof String) {
			String nome = (String) perda;
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
				"\nTipo de dado \"" + perda.getClass().getTypeName() + "\" não suportado."
			);
		}
		
	}

    /**
     * Converte o inicializador recebido em uma instância de inicializador.
     * @param inicializador tipo de inicializador.
     * @return instância do inicializador lido.
     */
	public Inicializador getInicializador(Object inicializador) {
		utils.validarNaoNulo(inicializador, "Inicializador nulo.");

		if (inicializador instanceof Inicializador) {
			return (Inicializador) inicializador;
		
		} else if (inicializador instanceof String) {
			String nome = (String) inicializador;
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
				"\nTipo de dado \"" + inicializador.getClass().getTypeName() + "\" não suportado."
			);
		}

	}
}
