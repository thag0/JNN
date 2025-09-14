package testes;

import java.text.DecimalFormat;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import jnn.Funcional;
import jnn.camadas.Camada;
import jnn.camadas.Conv2D;
import jnn.camadas.MaxPool2D;
import jnn.core.OpTensor;
import jnn.core.Utils;
import jnn.core.tensor.Tensor;
import jnn.modelos.Sequencial;
import jnn.serializacao.Serializador;
import lib.ged.Ged;
import lib.geim.Geim;

public class Playground {
	static Ged ged = new Ged();
	static OpTensor optensor = new OpTensor();
	static Geim geim = new Geim();
	static Utils utils = new Utils();
	static DecimalFormat df = new DecimalFormat();
	static Serializador serial = new Serializador();
	static Funcional jnn = new Funcional();

	public static void main(String[] args) {
		ged.limparConsole();

		Sequencial model = serial.lerSequencial("dados\\modelos\\modelo-treinado.nn");
		modelBenchmark(model);
	}

	public static void modelBenchmark(Sequencial model) {
		Camada[] camadas = model.camadas();

		Tensor x = new Tensor(model.camada(0).shapeEntrada());
		x.aplicar(_ -> randn());
		Tensor g = new Tensor(model.camadaSaida().shapeSaida());
		g.aplicar(_ -> randn());

		long t, total = 0;
		List<Map.Entry<String, Long>> res1 = new ArrayList<>();
		List<Map.Entry<String, Long>> res2 = new ArrayList<>();

		for (int i = 0; i < camadas.length; i++) {
			Camada c = camadas[i];

			t = System.nanoTime();
			x = c.forward(x);
			t = System.nanoTime() - t;

			res1.add(new AbstractMap.SimpleEntry<>(c.id + " - " + c.nome(), t));
			total += t;
		}
		res1.add(new AbstractMap.SimpleEntry<>("Total", total));
		total = 0;

		for (int i = camadas.length-1; i >= 0; i--) {
			Camada c = camadas[i];

			t = System.nanoTime();
			g = c.backward(g);
			t = System.nanoTime() - t;
			total += t;

			res2.add(new AbstractMap.SimpleEntry<>(c.id + " - " + c.nome(), t));
		}
		res2.add(new AbstractMap.SimpleEntry<>("Total", total));

		res1.sort((a, b) -> Long.compare(b.getValue(), a.getValue()));
		res2.sort((a, b) -> Long.compare(b.getValue(), a.getValue()));

		System.out.println("Tempos forward em ordem decrescente:");
		for (Map.Entry<String, Long> r : res1) {
			System.out.println(r.getKey() + ":\t" + formatarDecimal(r.getValue()) + " ns");
		}
		
		System.out.println();
		System.out.println("Tempos backward em ordem decrescente:");
		for (Map.Entry<String, Long> r : res2) {
			System.out.println(r.getKey() + ":\t" + formatarDecimal(r.getValue()) + " ns");
		}
	}

	public static String formatarDecimal(Number x) {
		return new DecimalFormat("#,###").format(x);
	}

	/**
	 * Gera um valor aleatório.
	 * @return valor aleatório.
	 */
	static double randn() {
		return Math.random()*2-1;
	}

	static void testeGradMaxPool() {
		Tensor x = new Tensor(1, 10, 10);
		x.preencherContador(true);

		MaxPool2D mp = new MaxPool2D(new int[]{2, 2});
		mp.construir(x.shape());

		Tensor g = new Tensor(mp.shapeSaida());
		g.preencherContador(false);		
		
		mp.forward(x);
		mp.backward(g);
	
		Tensor esperado = new Tensor(new double[][][]{
			{
				{0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0},
				{0.0, 24.0,  0.0, 23.0,  0.0, 22.0,  0.0, 21.0,  0.0, 20.0},
				{0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0},
				{0.0, 19.0,  0.0, 18.0,  0.0, 17.0,  0.0, 16.0,  0.0, 15.0},
				{0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0},
				{0.0, 14.0,  0.0, 13.0,  0.0, 12.0,  0.0, 11.0,  0.0, 10.0},
				{0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0},
				{0.0,  9.0,  0.0,  8.0,  0.0,  7.0,  0.0,  6.0,  0.0,  5.0},
				{0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0},
				{0.0,  4.0,  0.0,  3.0,  0.0,  2.0,  0.0,  1.0,  0.0,  0.0}
			}
		});
	
		System.out.println(esperado.equals(mp.gradEntrada()));
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
			 1,  2,
			-1,  0,
			 0, -1, 
			 2,  1,
		};

		Conv2D conv = new Conv2D(new int[]{1, 3, 3}, 2, new int[]{2, 2}, "linear");
		conv.kernel().copiarElementos(exemploFiltro);

		Tensor amostra = new Tensor(exemploEntrada);
		amostra.unsqueeze(0);
		
		Tensor prev = conv.forward(amostra);
		System.out.println(conv._buffer);
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
		optensor.conv2DFull(t1, t2, t3);
				
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