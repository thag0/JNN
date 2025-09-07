package testes;

import java.util.Random;
import java.util.concurrent.TimeUnit;

import jnn.camadas.Conv2D;
import jnn.core.OpTensor;
import jnn.core.tensor.Tensor;
import jnn.inicializadores.GlorotNormal;
import jnn.inicializadores.GlorotUniforme;
import jnn.inicializadores.Inicializador;
import jnn.inicializadores.Zeros;
import lib.ged.Ged;


public class Benchmark {
	static OpTensor optensor = new OpTensor();

	public static void main(String[] args){
		Ged ged = new Ged();
		ged.limparConsole();

		int[] shapeEntrada = {16, 26, 26};
		int[] shapeFiltro = {3, 3};
		int filtros = 20;

		// copiar entradas: 163.800 ns
		// copiar kernels: 569.700 ns
		// copiar saidas: 262.100 ns
		// correlações: 15.419.100 ns
		// add bias: 2.637.300 ns
		// Config conv = [
		//    entrada: (16, 26, 26)
		//    filtros: (20, 16, 3, 3)
		//    saida: (20, 24, 24)
		//    Tempo forward: 42ms
		// ]

		convForward(shapeEntrada, shapeFiltro, filtros);
		// testarForward();
		convBackward(shapeEntrada, shapeFiltro, filtros);
		// testarBackward();

		// Tensor a = new Tensor(28, 28);
		// Tensor b = new Tensor(3, 3);
		// Tensor c = new Tensor(26, 26);

		// long t = medirTempo(() -> optensor.correlacao2D(a, b, c));
		// System.out.println("tempo: " + new DecimalFormat().format(t) + " ns");
	}

	/**
	 * Gera um número aleatório no intervalo [-1, 1]
	 * @return valor aleatório.
	 */
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
			formatoEntrada, filtros, formatoFiltro, ativacao, iniKernel, iniBias
		);
		conv.inicializar();

		Tensor entrada = new Tensor(conv.shapeEntrada());
		entrada.preencherContador(true);

		long tempo = 0;
		tempo = medirTempo(() -> conv.forward(entrada));
	
		//resultados
		StringBuilder sb = new StringBuilder();
		String pad = "   ";

		sb.append("Config conv = [\n");
			sb.append(pad).append("entrada: " + conv._entrada.shapeStr() + "\n");
			sb.append(pad).append("filtros: " + conv._kernel.shapeStr() + "\n");
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
			formatoEntrada, filtros, formatoFiltro, ativacao, iniKernel, iniBias
		);
		conv.inicializar();

		//preparar dados pra retropropagar
		long randSeed = 99999;
		Random rand = new Random(randSeed);
		Tensor amostra = new Tensor(conv.shapeEntrada());
		amostra.aplicar((_) -> rand.nextDouble());
		conv.forward(amostra);

		Tensor grad = new Tensor(conv._gradSaida.shape());
		grad.aplicar((_) -> rand.nextDouble());

		long tempo = medirTempo(() -> conv.backward(grad));

		//resultados
		StringBuilder sb = new StringBuilder();
		String pad = "   ";

		sb.append("Config conv = [\n");
			sb.append(pad).append("entrada: " + conv._entrada.shapeStr() + "\n");
			sb.append(pad).append("filtros: " + conv._kernel.shapeStr() + "\n");
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
		Conv2D conv = new Conv2D(formEntrada, 16, new int[]{3, 3}, "linear", iniKernel, iniBias);
		conv.inicializar();

		Tensor entrada = new Tensor(conv.shapeEntrada());
		entrada.aplicar((_) -> Math.random());

		//simulação de propagação dos dados numa camada convolucional sem bias
		Tensor filtros = new Tensor(conv.kernel());
		Tensor saidaEsperada = new Tensor(conv.shapeSaida());
		
		//conv forward single thread
		int[] shapeK = filtros.shape();
		int[] shapeS = saidaEsperada.shape();

		int profEntrada = shapeK[1];
		int numFiltros = shapeK[0];

		int altSaida = shapeS[1];
		int largSaida = shapeS[2];

		for (int f = 0; f < numFiltros; f++) {
			Tensor kernel3D = filtros.subTensor(f);
			for (int e = 0; e < profEntrada; e++) {
				Tensor entrada2d = entrada.subTensor(e);
				Tensor kernel2D = kernel3D.subTensor(e);
				Tensor res = optensor.corr2D(entrada2d, kernel2D);
				
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
		int[] formEntrada = {10, 16, 16};
		Inicializador iniKernel = new GlorotUniforme(12345);
		Inicializador iniBias = new Zeros();
		Conv2D conv = new Conv2D(formEntrada, 16, new int[]{3, 3}, "linear", iniKernel, iniBias);
		conv.inicializar();

		Tensor amostra = new Tensor(conv.shapeEntrada());
		amostra.aplicar(_ -> randn());
		conv.forward(amostra);

		Tensor convE = conv._entrada.clone();
		Tensor convK = conv._kernel.clone();
		Tensor convGK = conv._gradKernel.clone();
		Tensor convGE = conv._gradEntrada.clone();

		Tensor gradiente = new Tensor(conv.shapeSaida());
		gradiente.aplicar(_ -> randn());

		//testar gradiente de entrada
		for (int i = 0; i < 2; i++) {
			conv.backward(gradiente);
			convGE.zero();
			convBackward(convE, convK, gradiente, convGK, convGE);
		}

		boolean kEsperado = convGK.comparar(conv._gradKernel);
		boolean gEsperado = convGE.comparar(conv._gradEntrada);
		if (kEsperado && gEsperado) {
			System.out.println("backward esperado: " + (kEsperado && gEsperado));
		} else {
			System.out.println("backward inesperado: gradK: " + kEsperado + " gradE: " + gEsperado);
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

		// implementação antiga

		for (int f = 0; f < filtros; f++) {
			for (int e = 0; e < entradas; e++) {
				// gradiente dos kernels
				Tensor entrada2D = entrada.slice(new int[]{e, 0, 0}, new int[]{e+1, altE, largE});
				entrada2D.squeeze(0);//3d -> 2d

				Tensor gradSaida2D = gradS.slice(new int[]{f, 0, 0}, new int[]{f+1, altS, largS});
				gradSaida2D.squeeze(0);//3d -> 2d

				Tensor resCorr = optensor.corr2D(entrada2D, gradSaida2D);
				resCorr.unsqueeze(0).unsqueeze(0);
				gradK.slice(new int[]{f, e, 0, 0}, new int[]{f+1, e+1, altF, largF}).add(resCorr);
			
				// gradientes das entradas
				Tensor kernel2D = kernel.slice(new int[]{f, e, 0, 0}, new int[]{f+1, e+1, altF, largF});
				kernel2D.squeeze(0).squeeze(0);
				Tensor resConv = optensor.conv2DFull(gradSaida2D, kernel2D);
				gradE.slice(new int[]{e, 0, 0}, new int[]{e+1, altE, largE}).squeeze(0).add(resConv);
			}
		}
	}

}
