package testes;

import java.util.Random;
import java.util.concurrent.TimeUnit;

import java.text.DecimalFormat;

import jnn.camadas.Conv2D;
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

		// int[] formEntrada = {16, 26, 26};
		// int[] formFitlro = {3, 3};
		// int filtros = 20;

		// convForward(formEntrada, formFitlro, filtros);
		// testarForward();
		// convBackward(formEntrada, formFitlro, filtros);
		// testarBackward();

		Tensor a = new Tensor(28, 28);
		Tensor b = new Tensor(3, 3);
		Tensor c = new Tensor(26, 26);

		long t = medirTempo(() -> optensor.correlacao2D(a, b, c));
		System.out.println("tempo: " + new DecimalFormat().format(t) + " ns");//220.000~s240.000 ns
	}

	static double randn() {
		return Math.random()*2-1;
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

		Conv2D conv = new Conv2D(
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

		sb.append("Config conv = [\n");
			sb.append(pad).append("entrada: " + conv._entrada.shapeStr() + "\n");
			sb.append(pad).append("filtros: " + conv._filtros.shapeStr() + "\n");
			sb.append(pad).append("saida: " + conv._saida.shapeStr() + "\n");
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

		Conv2D conv = new Conv2D(
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

		sb.append("Config conv = [\n");
			sb.append(pad).append("entrada: " + conv._entrada.shapeStr() + "\n");
			sb.append(pad).append("filtros: " + conv._filtros.shapeStr() + "\n");
			sb.append(pad).append("saida: " + conv._saida.shapeStr() + "\n");
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
	static void testarForward() {
		int[] formEntrada = {12, 16, 16};
		Inicializador iniKernel = new GlorotUniforme(12345);
		Inicializador iniBias = new Zeros();
		Conv2D conv = new Conv2D(formEntrada, new int[]{3, 3}, 16, "linear", iniKernel, iniBias);
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

	/**
	 * Testar com multithread
	 */
	static void testarBackward() {
		int[] formEntrada = {16, 16, 16};
		Inicializador iniKernel = new GlorotUniforme(12345);
		Inicializador iniBias = new Zeros();
		Conv2D conv = new Conv2D(formEntrada, new int[]{3, 3}, 16, "linear", iniKernel, iniBias);
		conv.inicializar();

		Tensor amostra = new Tensor(conv.formatoEntrada());
		amostra.aplicar(x -> randn());
		conv.forward(amostra);

		Tensor convE = conv._entrada.clone();
		Tensor convK = conv._filtros.clone();
		Tensor convGK = conv._gradFiltros.clone();
		Tensor convGE = conv._gradEntrada.clone();

		Tensor gradiente = new Tensor(conv.formatoSaida());
		gradiente.aplicar(x -> randn());

		//testar gradiente de entrada
		for (int i = 0; i < 2; i++) {
			convGE.preencher(0);
			conv.backward(gradiente);
			convBackward(convE, convK, gradiente, convGK, convGE);
		}

		boolean kEsperado = convGK.comparar(conv._gradFiltros);
		boolean gEsperado = convGE.comparar(conv._gradEntrada);
		if (kEsperado && gEsperado) {
			System.out.println("backward esperado : " + (kEsperado && gEsperado));
		} else {
			System.out.println("backward inesperado : gradK: " + kEsperado + " gradE: " + gEsperado);
		}
	}

	/**
	 * Apenas para testes
	 * @param entrada
	 * @param kernel
	 * @param gradS
	 * @param gradK
	 * @param gradE
	 */
	static void convBackward(Tensor entrada, Tensor kernel, Tensor gradS, Tensor gradK, Tensor gradE) {
		int[] shapeE = entrada.shape();
		int[] shapeK = kernel.shape();
		int[] shapeS = gradS.shape();

		final int filtros = shapeK[0];
		final int entradas = shapeK[1];

		final int altE = shapeE[1];
		final int largE = shapeE[2];
		final int altF = shapeK[2];
		final int largF = shapeK[3];
		final int altS = shapeS[1];
		final int largS = shapeS[2];

		for (int f = 0; f < filtros; f++) {
			for (int e = 0; e < entradas; e++) {
				// gradiente dos kernels
				Tensor entrada2D = entrada.slice(new int[]{e, 0, 0}, new int[]{e+1, altE, largE});
				entrada2D.squeeze(0);//3d -> 2d

				Tensor gradSaida2D = gradS.slice(new int[]{f, 0, 0}, new int[]{f+1, altS, largS});
				gradSaida2D.squeeze(0);//3d -> 2d

				Tensor resCorr = optensor.correlacao2D(entrada2D, gradSaida2D);
				resCorr.unsqueeze(0).unsqueeze(0);
				gradK.slice(new int[]{f, e, 0, 0}, new int[]{f+1, e+1, altF, largF}).add(resCorr);
			
				// gradientes das entradas
				Tensor kernel2D = kernel.slice(new int[]{f, e, 0, 0}, new int[]{f+1, e+1, altF, largF});
				kernel2D.squeeze(0).squeeze(0);
				Tensor resConv = optensor.convolucao2DFull(gradSaida2D, kernel2D);
				gradE.slice(new int[]{e, 0, 0}, new int[]{e+1, altE, largE}).squeeze(0).add(resConv);
			}
		}
	}

}
