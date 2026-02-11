import ged.Ged;
import jnn.core.tensor.Tensor;

public class Teste {

    public static void main(String[] args) {
        new Ged().limparConsole();
        
        float[][] arr = {
            {1, 2},
            {3, 4}
        };

        Tensor t = new Tensor(arr);
        t.print();
    }
    
}