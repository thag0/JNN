package testes;

import jnn.core.OpArray;
import jnn.core.OpMatriz;
import jnn.core.Utils;
import jnn.core.tensor.OpTensor4D;
import jnn.core.tensor.Tensor4D;
import jnn.modelos.Sequencial;
import jnn.serializacao.Serializador;
import lib.ged.Ged;
import lib.geim.Geim;

import java.util.concurrent.TimeUnit;

public class Playground{
	static Ged ged = new Ged();
	static OpArray oparr = new OpArray();
	static OpMatriz opmat = new OpMatriz();
	static OpTensor4D optensor = new OpTensor4D();
	static Geim geim = new Geim();
	static Utils utils = new Utils();

	public static void main(String[] args){
		ged.limparConsole();

		Sequencial modelo = new Serializador().lerSequencial("./dados/modelos/mlp-mnist-89.nn");
		Tensor4D amostra = new Tensor4D(modelo.camada(0).formatoEntrada());
		amostra.aplicar(x -> Math.random()*2-1);

		Tensor4D prev = modelo.forward(amostra);
		System.out.println(prev.reshape(10, 1).argmax());
	}

    /**
     * Testes
     */
	static void testeConv2dFull(){
		double[][] a = {
			{1, 6, 2},
			{5, 3, 1},
			{7, 0, 4},
		};
		double[][] b = {
			{1, 2},
			{-1, 0},
		};

		Tensor4D t1 = new Tensor4D(a);
		Tensor4D t2 = new Tensor4D(b);
		Tensor4D t3 = new Tensor4D(t1.dim3()+t2.dim3()-1, t1.dim4()+t2.dim4()-1);
		long tempo = medirTempo(
			() -> optensor.convolucao2DFull(t1, t2, t3, new int[]{0, 0}, new int[]{0, 0}, new int[]{0, 0})
		);
		
		Tensor4D esperado = new Tensor4D(new double[][]{
			{1, 8, 14, 4},
			{4, 7, 5, 2},
			{2, 11, 3, 8},
			{-7, 0, -4, 0}
		});

		t3.print();
		System.out.println("Resultado esperado: " + t3.equals(esperado));
		System.out.println("Tempo: " + TimeUnit.NANOSECONDS.toMillis(tempo) + "ms");
	}

    /**
     * Mede o tempo de execução da função fornecida.
     * @param func função.
     * @return tempo em nanosegundos.
     */
	static long medirTempo(Runnable func){
		long t = System.nanoTime();
		func.run();
		return System.nanoTime() - t;
	}

}