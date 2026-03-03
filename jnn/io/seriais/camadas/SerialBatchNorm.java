package jnn.io.seriais.camadas;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jnn.camadas.BatchNorm2D;
import jnn.core.tensor.Tensor;
import jnn.io.seriais.SerialBase;
import jnn.io.seriais.SerializadorCamada;

public class SerialBatchNorm extends SerialBase implements SerializadorCamada<BatchNorm2D> {

    @Override
    public void serializar(BatchNorm2D camada, DataOutputStream dos) throws IOException {
        escrever(dos, camada.nome());

        escrever(dos, camada.shapeIn());
        escrever(dos, camada.eps());
        escrever(dos, camada.momentum());

        Tensor[] params = camada.params();
        escrever(dos, params[0].data().paraArray());//gamma
        escrever(dos, params[1].data().paraArray());//beta
    
        escrever(dos, camada._runningMean.data().paraArray());
        escrever(dos, camada._runningVar.data().paraArray());
    }

    @Override
    public BatchNorm2D ler(DataInputStream dis) throws IOException {
        int[] shapeIn = lerArrInt(dis);
        float eps = lerFloat(dis);
        float momentum = lerFloat(dis);

        float[] gamma = lerArrFloat(dis);
        float[] beta = lerArrFloat(dis);
        float[] runMean = lerArrFloat(dis);
        float[] runVar = lerArrFloat(dis);
    
        BatchNorm2D bn = new BatchNorm2D(momentum, eps);
        bn.construir(shapeIn);

        Tensor[] params = bn.params();
        params[0].copiarElementos(gamma);
        params[1].copiarElementos(beta);
    
        bn._runningMean.copiarElementos(runMean);
        bn._runningVar.copiarElementos(runVar);

        return bn;
    }

    @Override
    public String nome() {
        return "batchnorm2d";
    }

    @Override
    public Class<BatchNorm2D> tipo() {
        return BatchNorm2D.class;
    }
    
}
