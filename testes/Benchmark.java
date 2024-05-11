package testes;

import java.util.Random;
import java.util.concurrent.TimeUnit;

import jnn.camadas.Convolucional;
import jnn.core.tensor.OpTensor;
import jnn.core.tensor.Tensor;
import jnn.inicializadores.GlorotNormal;
import jnn.inicializadores.GlorotUniforme;
import jnn.inicializadores.Inicializador;
import jnn.inicializadores.Zeros;
import lib.ged.Ged;

public class Benchmark{
	static OpTensor optensor = new OpTensor();

	public static void main(String[] args){
		Ged ged = new Ged();
		ged.limparConsole();

		// int[] formEntrada = {1, 28, 28};
		// int[] formFiltro = {3, 3};
		// int numFiltros = 18;
		// Convolucional conv = new Convolucional(formEntrada, formFiltro, numFiltros, "leaky-relu");

		// Tensor4D entrada = new Tensor4D(conv.entrada.shape());
		// entrada.map(x -> Math.random());

		// long tempo = 0;
		// tempo = medirTempo(() -> conv.forward(entrada));
		
		// System.out.println("Tempo: " + TimeUnit.NANOSECONDS.toMillis(tempo) + " ms");

		// --------------------------------------------------------
		int[] formEntrada = {16, 26, 26};
		int[] formFitlro = {3, 3};
		int filtros = 20;

		convForward(formEntrada, formFitlro, filtros);// 16ms ~ 30ms
		testarForward();
		// convBackward(formEntrada, formFitlro, filtros);// 38ms ~ 54ms
	}

	/**
	 * Calcula o tempo de propagação direta da camada convolucional.
	 * <p>
	 *    Os valores são arbritários e podem ser ajustados para testar
	 *    diferentes cenários de estresse da camada.
	 * </p>
	 */
	static void convForward(int[] formatoEntrada, int[] formatoFiltro, int filtros){
		String ativacao = "sigmoid";

		final long seed = 123456789;
		final Inicializador iniKernel = new GlorotNormal(seed);
		final Inicializador iniBias = new GlorotNormal(seed);

		Convolucional conv = new Convolucional(
			formatoEntrada, formatoFiltro, filtros, ativacao, iniKernel, iniBias
		);
		conv.inicializar();

		Tensor entrada = new Tensor(conv.formatoEntrada());
		entrada.preencherContador(true);

		long tempo = 0;
		tempo = medirTempo(() -> conv.forward(entrada));
	
		//resultados
		StringBuilder sb = new StringBuilder();
		String pad = "   ";

		int[] e = conv.formatoEntrada();
		int[] s = conv.formatoSaida();

		sb.append("Config conv = [\n");
			sb.append(pad).append("filtros: " + conv.kernel().shapeStr() + "\n");
			sb.append(pad).append("entrada: (" + e[0] + ", " + e[1] + ", " + e[2] + ")\n");
			sb.append(pad).append("saida: (" + s[0] + ", " + s[1] + ", " + s[2] + ")\n");
			sb.append(pad).append("Tempo forward: " + TimeUnit.NANOSECONDS.toMillis(tempo) + "ms\n");
		sb.append("]\n");

		System.out.println(sb.toString());
	}

	/**
	 * Calcula o tempo de retropropagação da camada convolucional.
	 * <p>
	 *    Os valores são arbritários e podem ser ajustados para testar
	 *    diferentes cenários de estresse da camada.
	 * </p>
	 */
	static void convBackward(int[] formatoEntrada, int[] formatoFiltro, int filtros){
		String ativacao = "sigmoid";

		final long seed = 123456789;
		final Inicializador iniKernel = new GlorotNormal(seed);
		final Inicializador iniBias = new GlorotNormal(seed);

		Convolucional conv = new Convolucional(
			formatoEntrada, formatoFiltro, filtros, ativacao, iniKernel, iniBias
		);
		conv.inicializar();

		//preparar dados pra retropropagar
		long randSeed = 99999;
		Random rand = new Random(randSeed);
		Tensor amostra = new Tensor(conv.formatoEntrada());
		amostra.aplicar((x) -> rand.nextDouble());
		conv.forward(amostra);

		Tensor grad = new Tensor(conv._gradSaida.shape());
		grad.aplicar((x) -> rand.nextDouble());

		long tempo = medirTempo(() -> conv.backward(grad));

		//resultados
		StringBuilder sb = new StringBuilder();
		String pad = "   ";

		int[] e = conv.formatoEntrada();
		int[] s = conv.formatoSaida();

		sb.append("Config conv = [\n");
			sb.append(pad).append("filtros: " + conv.kernel().shapeStr() + "\n");
			sb.append(pad).append("entrada: (" + e[0] + ", " + e[1] + ", " + e[2] + ")\n");
			sb.append(pad).append("saida: (" + s[0] + ", " + s[1] + ", " + s[2] + ")\n");
			sb.append(pad).append("Tempo backward: " + TimeUnit.NANOSECONDS.toMillis(tempo) + "ms\n");
		sb.append("]\n");

		System.out.println(sb.toString());
	}

	/**
	 * Calcula o tempo de execução (nanosegundos) de uma função
	 * @param func função desejada.
	 * @return tempo de processamento.
	 */
	static long medirTempo(Runnable func){
		long t = System.nanoTime();
		func.run();
		return System.nanoTime() - t;
	}

	/**
	 * Testar com multithread
	 */
	static void testarForward(){
		int[] formEntrada = {12, 16, 16};
		Inicializador iniKernel = new GlorotUniforme(12345);
		Inicializador iniBias = new Zeros();
		Convolucional conv = new Convolucional(formEntrada, new int[]{3, 3}, 16, "linear", iniKernel, iniBias);
		conv.inicializar();

		Tensor entrada = new Tensor(conv.formatoEntrada());
		entrada.aplicar((x) -> Math.random());

		//simulação de propagação dos dados numa camada convolucional sem bias
		Tensor filtros = new Tensor(conv.kernel());
		Tensor saidaEsperada = new Tensor(conv.formatoSaida());
		
		//conv forward single thread
		int[] shapeE = entrada.shape();
		int[] shapeK = filtros.shape();
		int[] shapeS = saidaEsperada.shape();

		int profEntrada = shapeK[1];
		int altEntrada = shapeE[1];
		int largEntrada = shapeE[2];
		int numFiltros = shapeK[0];
		int altKernel = shapeK[2];
		int largKernel = shapeK[3];
		
		int altSaida = shapeS[1];
		int largSaida = shapeS[2];

		for (int f = 0; f < numFiltros; f++){
			for (int e = 0; e < profEntrada; e++) {
				Tensor entrada2d = entrada.slice(new int[]{e, 0, 0}, new int[]{e+1, altEntrada, largEntrada});
				entrada2d.squeeze(0);

				Tensor kernel2D = filtros.slice(new int[]{f, e, 0, 0}, new int[]{f+1, e+1, altKernel, largKernel});
				kernel2D.squeeze(0);
				kernel2D.squeeze(0);

				Tensor res = optensor.correlacao2D(entrada2d, kernel2D);
				res.unsqueeze(0);
				saidaEsperada.slice(new int[]{f, 0, 0}, new int[]{f+1, altSaida, largSaida}).add(res);
			}
		}

		Tensor prev = conv.forward(entrada);

		System.out.println("Forward esperado: " + prev.comparar(saidaEsperada));
	}

}
