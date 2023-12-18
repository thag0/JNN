# Biblioteca de Redes Neurais

Estou adaptando meu modelo de MLP para o formato matricial, com o objetivo de tornar o modelo mais modular para aplicações futuras.

Nesse modelo o centro de tudo é a classe matriz que realiza as operações mais importantes dentro da rede, com isso quero tornar o desempenho
da rede melhor, tentando paralelizar o uso das operações matriciais que são usadas a todo momento durante a execução do programa.

Estou conseguindo expandir a modularização da rede, criando uma camada base que é herdada para as novas camadas criadas, ela possui a base das 
implementações necessárias pro funcionamento de uma camada dentro dos modelos.

Depois de mexer bastante e estudar um pouco, consegui fazer a implementação da camada convolucional, ela ainda ta sendo testada, ainda mais na 
compatibilidade com os otimizadores, mas os resultados estão indo bem. Consegui testar com um exemplo básico do MNIST com 10 digitos de 0 a 9, só
pra ver se ela já conseguia pelo menos ter overfitting sobre os dados, e deu certo.

Também estou tentando criar a api de um modelo sequencial para empilhar camadas dentro de um modelo, até o momento não estou tendo tantos problemas.

# Implementação do modelo Sequencial

![treino sequencial](https://github.com/thag0/Biblioteca-de-Redes-Neurais/assets/91092364/7fe7881b-5f7c-4e69-a387-418705667b48)
