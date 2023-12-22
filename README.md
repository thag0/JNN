# Biblioteca de Redes Neurais

Estou adaptando meu modelo de MLP para o formato matricial, com o objetivo de tornar o modelo mais modular para aplicações futuras.

Nesse modelo o centro de tudo é a classe matriz que realiza as operações mais importantes dentro da rede, com isso quero tornar o desempenho
da rede melhor, tentando paralelizar o uso das operações matriciais que são usadas a todo momento durante a execução do programa.

Estou conseguindo expandir a modularização da rede, criando uma camada base que é herdada para as novas camadas criadas, ela possui a base das 
implementações necessárias pro funcionamento de uma camada dentro dos modelos.

Depois de mexer bastante e estudar um pouco, consegui fazer a implementação da camada convolucional, ela ainda ta sendo testada, ainda mais na 
compatibilidade com os otimizadores, mas os resultados estão indo bem. Consegui testar com um exemplo básico do MNIST com 10 digitos de 0 a 9, só
pra ver se ela já conseguia pelo menos ter overfitting sobre os dados, e deu certo.

# Implementação do modelo Sequencial

Estou criando e implementando uma api para um modelo sequencial de camadas, ele é uma generalização de um modelo base, assim como na api "RedeNeural", ambos possuem as mesmas implementações de métodos, mas o modelo sequencial tem suas especialidades.

- O modelo sequencial não é limitado apenas à camadas Densas, podendo empilhar camadas convolucionais, flatten e maxpooling (até o momento foi o que adicionei).
- O modelo sequencial é capaz de lidar com diferentes tipos de dados de entrada e saída, dependendo das camadas configuradas para ele.

Consegui criar essa generalização para o modelo sequencial criando a classe mãe "Camada" que possui as implementações de métodos necessárias para o funcionamento do modelo. Novas camadas devem implementar esses métodos e seguir os padrões que as demais camadas seguem para manter o funcionamento correto de construção, propagação e treinamento das camadas.

Seguindo a ideia de generalização, os otimizadores também sofreram algumas mudanças para poder lidar com multiplos tipos de camadas. Agora lidam com formas vetorizadas dos kernels e bias das camadas, sem se preocupar com a formatação de cada uma em especifico.

![treino sequencial](https://github.com/thag0/Biblioteca-de-Redes-Neurais/assets/91092364/7fe7881b-5f7c-4e69-a387-418705667b48)

*Exemplo de treino usando o modelo sequencial, usando o otimizador sgd e função de perda mse*

Até o momento o modelo sequencial está lidando perfeitamente bem com os métodos já criados na primeira api "RedeNeural" as generalizações estão funcionando bem. 

As dificuldades no momento estão sendo lidar com modelos convolucionais, principalmente na questão de treinamento, já que eles não estão aprendendo e parecem ficar presos em mínimos locais rapidamente.

# Treinamento de modelos Convolucionais

![treino conv](https://github.com/thag0/Biblioteca-de-Redes-Neurais/assets/91092364/088b40d8-bf53-491e-897f-7138f1c5ea88)

Estou iniciando os teste com modelos convolucionais e já obtive ótimos resultados usando um pequeno conjunto de dados do mnist pra treinar um modelo simples.