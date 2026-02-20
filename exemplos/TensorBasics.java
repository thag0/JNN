package exemplos;

import ged.Ged;
import jnn.core.tensor.Tensor;

/**
 * A
 */
@SuppressWarnings("unused")
public class TensorBasics {
    static Ged ged = new Ged();

    public static void main(String[] args) {
        ged.limparConsole();
        
        // Criação de um Tensor
        // 1 - A partir de um shape, com seu conteúdo zerado.
        Tensor a = new Tensor(3);

        // 2 - A partir de um array primitivo do java
        float[][] arr = {
            {1, 2},
            {3, 4}
        };
        Tensor b = new Tensor(arr);// Dados do array são copiados para o Tensor.

        // 3 - A partir de outro Tensor
        Tensor c = new Tensor(b);// Dados do Tensor base são copiados para o novo.
    
        // Operações
        //      Muitas operações acontecem de forma "in-place", ou seja,
        //      o resultado é gravado dentro do tensor que realizou a
        //      chamada de função
        // 1 - Adição
        Tensor d = new Tensor(2, 2);
        Tensor e = new Tensor(2, 2);
        d.add(e);// Resultado armazenado em D.

        // Operações especiais, que alteram o formato do Tensor, atuam
        // retornando views, que são novos Tensores que compartilham os
        // mesmo dados do Tensor original, mas possuem estruturas diferentes.
        // 1 - Reshape
        Tensor f = new Tensor(3, 2, 2);
        Tensor g = f.reshape(2, 3, 2);//Importante ter a mesma quantidade de dados ao final.

        // 2 - Transpor
        Tensor h = new Tensor(3, 2);
        Tensor i = h.transpor();// Formato final (2, 3)

        // Transformações
        // Existem algumas operações que alteram o conteúdo do Tensor 
        // utilizando funções especiais a critério da necessidade do usuário.
        Tensor j = new Tensor(2, 2);
        Tensor k = j.aplicar(val -> val * 2);// Multiplica todos o valores do Tensor por 2.
        Tensor l = j.reduce(0, (val1, val2) -> val1 + val2);// Soma todos os valores do Tensor por redução.

        // Broadcasting
        // Algumas operações especiais podem ser realizadas entre dois tensores
        // se os shapes deles forem compatíveis para transmissão (broadcast).
        Tensor m = new Tensor(3, 2);
        Tensor n = new Tensor(1, 2);// formato diferente de M, mas compatível para broadcast
        Tensor o = m.broadcast(n, (_m, _n) -> _m * _m);// Formato final (3, 2)
    }
}
