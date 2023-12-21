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

# Dificuldades com o treinamento de modelos convolucionais

![problema conv](https://github.com/thag0/Biblioteca-de-Redes-Neurais/assets/91092364/f0cde31f-fd5b-4690-b477-913e04d436ed)

Atualmente o que to tentando entender e corrigir é o problema de treinamento nos modelos convolucionais.

Fiz o teste usando o dataset do mnist (que está no projeto) e percebo que o modelo não tem capacidade de reconhecer e aprender as características das imagens, já testei coisas como:
 - Leitura correta dos arquivos de imagens;
 - Transformação correta dos dados de imagem em dados de treinamento;
 - Testes com diferentes funções de ativação;
 - Testes com diferentes parâmetros de arquitetura (tamanho de filtros, número de filtro, tamanho de máscara de pooling, número neurônios);
 - Diferentes arquiteturas (pilha de camadas convolucionais, camada convolucional seguida de pooling, várias camadas densas)
 - Testes nos resultados internos das operações dentro da camada convolucional, como operação de correlação cruzada e convolução full;
 - Análise nas transformações dos dados entre camadas, como saída da camada Convolucional para entrada da camada Flatten e Saída da camada Flatten para a camada Densa;

Ainda não consegui chegar ao resultado do que pode estar causando esses problemas, mas de certeza já tenho que as camadas Densa e Flatten estão funcionado corretamente e trazendo os resultados esperados. A camada de pooling ainda foi pouco testada mas em cenários mais controlados já vi que ela entrega os resultados esperados, mas não consegui chegar num resultado certo ainda.