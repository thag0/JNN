package testes;

import jnn.camadas.Convolucional;
import jnn.core.OpArray;
import jnn.core.Utils;
import jnn.core.tensor.OpTensor;
import jnn.core.tensor.Tensor;
import lib.ged.Ged;
import lib.geim.Geim;

public class Playground {
	static Ged ged = new Ged();
	static OpArray oparr = new OpArray();
	static OpTensor optensor = new OpTensor();
	static Geim geim = new Geim();
	static Utils utils = new Utils();

	public static void main(String[] args) {
		ged.limparConsole();

		testeConv2dFull();
	}

	static double randn() {
		return Math.random()*2-1;
	}

	/**
	 * Testes
	 */
	static void testeConvForward() {
		double[][] exemploEntrada = {
			{1, 6, 2},
			{5, 3, 1},
			{7, 0, 4},
		};
		double[] exemploFiltro = {
			 1, 2, 
			-1, 0,
			 0, -1, 
			 2,  1,
		};
		
		Convolucional conv = new Convolucional(new int[]{1, 3, 3}, new int[]{2, 2}, 2, "linear");
		conv.kernel().copiarElementos(exemploFiltro);

		Tensor amostra = new Tensor(exemploEntrada);
		amostra.unsqueeze(0);

		Tensor prev = conv.forward(amostra);
		System.out.println(conv._somatorio);
		System.out.println(prev);

		Tensor resEsperado = new Tensor(
			new double[]{
				8, 7,
				4, 5,
				7, 5, 
				11, 3
			},
			2, 2, 2
		);

		System.out.println("Saída esperada: " + prev.comparar(resEsperado));
	}

	/**
     * Testes
     */
	static void testeConv2dFull() {
		double[][] a = {
			{1, 6, 2},
			{5, 3, 1},
			{7, 0, 4},
		};
		double[][] b = {
			{1, 2},
			{-1, 0},
		};

		Tensor t1 = new Tensor(a);
		Tensor t2 = new Tensor(b);
		Tensor t3 = new Tensor(4, 4);
		optensor.convolucao2DFull(t1, t2, t3);
				
		Tensor esperado = new Tensor(new double[][]{
			{1, 8, 14, 4},
			{4, 7, 5, 2},
			{2, 11, 3, 8},
			{-7, 0, -4, 0}
		});

		t3.print();
		System.out.println("Resultado esperado: " + t3.equals(esperado));
	}

    /**
     * Mede o tempo de execução da função fornecida.
     * @param func função.
     * @return tempo em nanosegundos.
     */
	static long medirTempo(Runnable func) {
		long t = System.nanoTime();
		func.run();
		return System.nanoTime() - t;
	}

}