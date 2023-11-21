# Rede-Neural-com-Matrizes

Estou adaptando meu modelo de MLP para o formato matricial, com o objetivo de tornar o modelo mais modular para aplicações futuras.

Nesse modelo o centro de tudo é a classe matriz que realiza as operações mais importantes dentro da rede, com isso quero tornar o desempenho
da rede melhor, tentando paralelizar o uso das operações matriciais que são usadas a todo momento durante a execução do programa.

Em termos de tempo de execução, essa abordagem é bem mais lenta que a anterior que usa arrays de neurônios e arrays de pesos/bias. Minha ideia inicial é melhorar o 
desempenho desse modelo de rede e depois começar a trabalhar no modelo convolucional, usando a CamadaDensa como mais um tipo de camada para o modelo.
