

import java.text.DecimalFormat;
import java.util.concurrent.TimeUnit;

import ged.Ged;
import jnn.camadas.Conv2D;
import jnn.camadas.Densa;
import jnn.camadas.Entrada;
import jnn.camadas.Flatten;
import jnn.camadas.pooling.MaxPool2D;
import jnn.core.JNNutils;
import jnn.core.tensor.Tensor;
import jnn.modelos.Sequencial;
import jnn.core.JNNnative;

@SuppressWarnings("unused")
public class TesteTempos {
    static final int WARMUP = 50;
    static final int REPETICOES = 10;
    
    public static void main(String[] args) {
        JNNnative.jni = true;// interface pro código C (melhor desempenho)
        JNNnative.setThreads(4);

        Ged ged = new Ged();
        // ged.limparConsole();
        
        int[] shapeIn = {3, 32, 32};
        Sequencial model = new Sequencial(
            new Entrada(shapeIn),
            new Conv2D(32, new int[]{3, 3}, "relu"),
            new MaxPool2D(new int[]{2, 2}),
            new Conv2D(64, new int[]{3, 3}, "relu"),
            new MaxPool2D(new int[]{2, 2}),
            new Conv2D(128, new int[]{3, 3}, "relu"),
            new Flatten(),
            new Densa(128, "relu"),
            new Densa(10, "softmax")
        );

        model.compilar("adam", "entropia-cruzada");
        // model.print();

        forward(model);
        forward(model);
        forward(model);

        backward(model);
        backward(model);
        backward(model);
        backward(model);
    }

    static void forward(Sequencial model) {
        int lotes = 32;
        int[] shapeIn = model.shapeIn();
        int[] shapeX = {lotes, shapeIn[0], shapeIn[1], shapeIn[2]};
        Tensor x = new Tensor(shapeX).aplicar(_ -> randn());

        // warmup
        for (int i = 0; i < WARMUP; i++) {
            model.forward(x);
        }

        long total = 0;
        for (int i = 0; i < REPETICOES; i++) {
            long t = System.nanoTime();
                Tensor y = model.forward(x);
            t = System.nanoTime() - t;
        
            total += t;
        }
    
        total /= REPETICOES;
        
        System.out.println("JNN forward: " + formatarDecimal(total) + " ns (" + TimeUnit.NANOSECONDS.toMillis(total) + " ms).");
    }

    static void backward(Sequencial model) {
        int lotes = 32;
        int[] shapeIn = model.shapeIn();
        int[] shapeX = {lotes, shapeIn[0], shapeIn[1], shapeIn[2]};
        Tensor x = new Tensor(shapeX).aplicar(_ -> randn());
        model.forward(x);// pra ter entradas válidas no modelo.

        int[] shapeG = model.shapeOut();
        Tensor g = new Tensor(shapeG).aplicar(_ -> randn());

        // warmup
        for (int i = 0; i < WARMUP; i++) {
            model.backward(g);
        }

        long total = 0;
        for (int i = 0; i < REPETICOES; i++) {
            long t = System.nanoTime();
                model.backward(g);
            t = System.nanoTime() - t;
        
            total += t;
        }
    
        total /= REPETICOES;
        System.out.println("JNN backward: " + formatarDecimal(total) + " ns (" + TimeUnit.NANOSECONDS.toMillis(total) + " ms).");
    }

	public static String formatarDecimal(Number x) {
		return new DecimalFormat("#,###").format(x);
	}

    static float randn() { 
        return JNNutils.randFloat();
    }
}
