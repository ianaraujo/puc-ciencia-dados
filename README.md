# Predição de Fraudes "Card Not Present" (IEEE-CIS Dataset)

O problema escolhido para realização do projeto foi de fraudes em transações sem cartão presencial, conhecidas como **"Card Not Present Fraud"**. Os dados foram preparados e disponibilizados pela IEEE Computational Intelligence Society e lançados durante uma competição [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/overview) no Kaggle.

O que torna esse projeto mais interessante é que os dados são de **transações reais** foram fornecidos pela [Vesta Corporation](https://vesta.io/). A Vesta é uma empresa especializada em soluções de proteção contra fraudes e processamento de pagamentos para transações móveis e online. A empresa utiliza modelos avançados de *machine learning* para analisar mais de US$ 4 bilhões em transações anualmente, fornecendo serviços que permitem a aprovação de vendas em milissegundos e o processamento de pagamentos em mais de 40 países.

O conjunto de dados possui cerca de 600.000 registros e contém mais de 430 características. Portanto, além do desafio de modelagem para predição de fraudes, é preciso considerar o grande volume de dados e a alta dimensionalidade. Por serem transações reais, um grande número de *features* foram anonimizadas, garantindo a privacidade dos clientes, portanto, não é possível saber qual o conteúdo real de todas as variáveis.

## Definição do Problema

Imagine a seguinte situação. Você compra um café da manhã na padaria e se dirige ao caixa para pagar. Como é muito comum nos tempos atuais, você utiliza seu telefone para pagar, usando um cartão digital por aproximação, mas sua compra é negada. Ou você decide aproveitar as promoções de fim de ano em determinada plataforma de e-commerce, preenche as informações do seu cartão, mas sua compra não pode ser efetuada.

Embora você tenha certeza que possui dinheiro suficiente para realizar a compra, por algum motivo sua transação não é efetuada. Isso acontece por que, todos os anos, mais de 30 bilhões de dólares são movimentados em transações fraudulentas de cartões de crédito e as empresas envolvidas nesses serviços, como bancos, instituições de pagamento e prestadoras de serviço, investem pesado em **sistemas de prevenção de fraude**. Apesar de nem sempre serem acertivos, como no caso descrito acima quando houve uma falso positivo, esses sistema podem evitar uma grande dor de cabeça para os clientes e para essas instituições.

A fraude em transações "Card Not Present" (CNP) ocorre quando compras são realizadas sem a presença física do cartão. Com o avanço da tecnologia e a popularização das compras pela internet, esse tipo de transação tornou-se predominante. No Brasil, por exemplo, 61% dos consumidores preferem comprar online em vez de em lojas físicas, e 78% realizam pelo menos uma compra mensal pela internet. Nessas compras, os consumidores inserem os dados do cartão, como número, data de validade e código de segurança, ou utilizam carteiras digitais em dispositivos móveis, facilitando as transações, mas também aumentando os riscos de fraude.

Considerando esse cenário, o objetivo do trabalho é criar um **modelo de classificação** capaz de estimar a probabilidade das trações serem verdadeiras ou fraudes. Em paralelo, também discutir conceitos importantes desse tipo de problema, como classes desbalanceadas, redução de dimensionalidade e uso adequado de métricas.

## Análise Exploratória de Dados

Como discutido, neste problema vamos criar um classificador para prever a probabilidade de uma transação ser fraudulenta, indicada pelo alvo binário `isFraud`. Os dados de treinamento do Kaggle estão divididos em dois arquivos, `train_identity.csv` e `train_transaction.csv`, que são unidos pela chave `TransactionID`.

Por ser uma competição, o Kaggle não permite baixar os arquivos sem fazer login, por isso, não foi possível ler os dados através da URL. Foi necessário baixar os arquivos, que podem ser salvos na pasta `/data` e ler localmente.

Por causa do grande volume de *features* nos dados de transações (394), vamos ler os dados em *chunks* ou blocos de 100.000 registros, o que otimiza a alocação de memória da máquina. Após concatenar os blocos em um único DataFrame, podemos deletar esses blocos para economizar recursos.

Em alguns casos específicos de modelagem preditiva, lidamos com um problema conhecido como “classes desbalanceadas”. Isso acontece quando queremos classificar determinado exemplo, seja em tarefas binárias ou de múltiplas classes, mas os dados de treinamento específicos não apresentam uma distribuição equilibrada dessas classes.

### isFraud

Em problemas de detecção de fraudes, assim como predição de CHURN ou doenças raras, o fenômeno que queremos prever, geralmente, é subrepresentado nos dados de treino. Como podemos ver, isso aconte em nosso conjunto de dados, uma vez que apenas 3,5% do dados de treino são positivos para fraude.

Quando há desbalanceamento de classes é necessário conhecer algumas técnicas para lidar com esse problemas de forma eficiente, que pode ser abordado pela via da reamostragem, buscando equilibrar as classes removendo exemplos da classe majoritária ou criando exemplos artificiais da classe minoritária. Outra abordagem é o uso correto de métricas de avaliação.

A acurácia é uma métrica muito usada em problemas gerais, mas que não tem grande utilidade para esse problema específico. Imagine que o modelo aprenda a dizer que todas as transações são verdadeiras, a acurácia seria próxima de 97%, mas nosso modelo seria muito ruim. Métricas mais adequadas são a **precisão** e o **recall**, que juntas formam o **F1 score**.

Essa métricas são calculadas apartir do número de falsos positivos ou falsos negativos e do verdadeiros positivos. Assim, podemos ajustar nosso modelo mais adequadamente para nosso objetivo alvo: minimizar falsos positivos (recall) ou minimizar falsos negativos (precisão). Vamos discutir essas técnicas mais profundamente ao longo do trabalho.

### TransactionAmt

A análise dos valores de transações revela uma distribuição altamente concentrada em valores baixos, com a maioria das transações ocorrendo abaixo de 200. Essa característica aponta para um desbalanceamento nos dados, com uma pequena proporção de transações de valores significativamente altos. Em termos de modelagem de machine learning, isso implica na necessidade de estratégias específicas para lidar com a amplitude dos dados e a presença de outliers. Transformações, como o uso de escalas logarítmicas, podem ser úteis para estabilizar a variância e melhorar o desempenho de modelos sensíveis à escala.

Além disso, a predominância de transações de baixo valor sugere que os modelos podem precisar de ajustes para evitar vieses que priorizem padrões comuns e negligenciem a detecção de eventos raros, como fraudes em transações de valores elevados. Avaliar o impacto dessas transações menos frequentes será crucial para garantir que o modelo seja robusto e eficaz em diferentes cenários.

![TransactionAmt](/images/transaction-amount.png)

A análise dos dados revela que transações fraudulentas possuem, em média, valores ligeiramente superiores às transações não fraudulentas (149,24 contra 134,51). No entanto, essa diferença de médias é relativamente pequena, e as distribuições de valores para ambos os casos mostram uma sobreposição significativa. Isso indica que o valor da transação, isoladamente, não é um bom discriminador para identificar fraudes.

Apesar disso, o valor da transação pode ser uma variável útil quando combinado com outros fatores, já que existe uma leve tendência de fraudes ocorrerem em transações com valores mais elevados. Para maximizar sua utilidade em um modelo de machine learning, seria interessante explorar interações com outras variáveis, realizar transformações como escalas logarítmicas e investigar a influência de outliers.

### Análise de Componentes Principais

Conforme apresentado no início do notebook, os dados de treino estão dividios em dois conjuntos, transações e identidade, que compartilham a chave `TransactionID`. É necessário realizar o *join* dessas duas tabelas, de modo que a máquina consiga processar esses relacionamentos sem exceder o limite de memória.

Além disso, por serem quase 600.000 registros, vamos criar uma amostra estratificada de 10% do conjunto original com objetivo de reduzir o tempo de treinamento de modelos, permitindo testar diversas combinações em menos tempo. Depois que estivermos satisfeitos com modelo escolhido e os parâmetros, podemos utilizar o *dataset* completo para treinar o modelo final.

A amostragem estratificada é uma técnica que permite criar subgrupos homogêneos (estratos) com base em características relevantes para o estudo. No nosso caso, é fundamental manter a proporção de fraudes e transações verdadeira ou poderíamos correr o risco de criar uma amostra aleatória em que o desbalanceamento das classes fosse ainda mais acentuado.

Outra categoria que deve ser levada em consideração ao realizar a amostra estratificada é a `hasIdentity`, que representa a preseção de dados correspondentes da tabela de identidade. Sem considerar essa variável, podemos criar uma amostra aleatória em que grande parte das transações tenham identidade correspondente, o que pode enviesar nosso modelo.

Analisando as *features* do nosso conjunto de dados, observamos que existem 339 variáveis que começam com a letra 'V' e são seguidas de um número, entre 1 e 339. Segundo a [descrição dos dados](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/101203) disponibilizada pelo Kaggle, essas são *features* criadas pela Vesta, empresa que disponibilizou o *dataset*, após um processo próprio de feature engineering. A descrição não fornece maiores detalhes sobre a criação dessas variáveis.

Por serem muito numerosas, o que exige maior poder computacional e pode dificultar a capacidade do modelo de generalizar bem os dados, vamos conduzir um processo de redução de dimensionalidade.

Para isso, utilizaremos a técnica de Análise de Componentes Principais (PCA), que permite transformar as features originais em um novo conjunto de variáveis não correlacionadas, chamadas de componentes principais. Esses componentes são ordenados de forma a capturar a maior variância possível dos dados nas primeiras dimensões, permitindo que descartemos as componentes menos relevantes. Assim, conseguimos reduzir a complexidade do modelo, preservando a maior parte da informação contida nas features originais.

![PCA](/images/pca.png)

O gráfico acima mostra a variância explicada acumulada em função do número de componentes principais após utilizar o PCA (Principal Component Analysis). Inicialmente, a variância explicada aumenta rapidamente com o acréscimo dos primeiros componentes. Isso indica que esses componentes capturam a maior parte da variância dos dados. Conforme mais componentes são adicionados, o crescimento da variância explicada desacelera, porque os componentes adicionais capturam porções cada vez menores da variância. Cerca de 50 a 60 componentes já explicam uma grande fração da variância total (provavelmente mais de 90%).

Sendo assim, podemos determinar uma fração aceitável da variância (95%, por exemplo) e determinar quantos componentes são necessários para capturar essa fração. Dessa forma, é possível reduzir o dataset de muitas dimensões para apenas `d`, otimizando o desempenho de algoritmos posteriores e os recursos necessários para treiná-los. O código abaixo encontra o valor de `d`, que será usada mais para frente durante o pré-processamento, para determinada variação *target*.

## Feature Engineering

O processo de *feature engineering* aplicado ao conjunto de dados foi realizado com o objetivo de criar novas variáveis mais informativas e tratar valores que poderiam impactar negativamente o desempenho do modelo.

Foi realizado um processo de criação de novas features utilizando a variável TransactionAmt (valor da transação) e as variáveis categóricas card1, card4, card6, P_emaildomain, R_emaildomain e addr1. Para cada uma dessas colunas, foram criadas as seguintes features derivadas: `amount_mean` e `amount_std`.

Essas variáveis têm o objetivo de capturar o quanto o valor de uma transação específica se desvia da média ou da dispersão geral das transações para cada categoria. Isso pode ajudar a identificar comportamentos atípicos. Uma transformação logarítmica também foi aplicada à variável `TransactionAmt`, criando a nova feature `TransactionAmtLog`. A transformação logarítmica é útil para reduzir a assimetria de distribuições muito enviesadas e facilitar a modelagem.

Outro passo importante foi a conversão da variável `TransactionDT`, que representa um timestamp relativo, em uma data real. Para isso, foi definida uma data de referência inicial (2022-01-01), e a partir do número de segundos em TransactionDT, foi calculada a data exata de cada transação. A partir dessa data, foram extraídas novas features de data e tempo: dia da semana, dia e hora.

O conjunto de dados final para treinamento do modelo possui 463 features (desconsiderando a variável target), entre as quais 29 foram criadas durante o processo de criação de features (*feature engineering*). Também devemos considerar que, ao realizar a análise de componentes principais, identificamos que 339 features podem ser representadas por aproximadament 90 componenetes principais, mantendo 95% da variância do conjunto de features original.

Portanto, mesmo com a redução significativa de dimensionalidade, nosso modelo ainda deve analisar cerca de 214 features, buscando encontrar padrões que ajudem a prever a ocorrência de fraudes.

## Pré-processamento

Apesar da maior parte das features ser númerica, algumas delas representar na verdade variáveis categóricas. Essa informação, de quais são as variáveis categóricas, foi disponibilizada pelos organizadores da competição no Kaggle. Assim, mesmo sem os rótulos de algumas delas (M1-M9), por questão de proteção de dados reais, ainda é possível ter a informação e tratar essas variáveis adequadamente, durante o pré-processamento.

Categorical Features (Transaction)

- ProductCD
- emaildomain
- card1 - card6
- addr1, addr2
- P_emaildomain
- R_emaildomain
- M1 - M9

Categorical Features (Identity)

- DeviceType
- DeviceInfo
- id_12 - id_38

Categorical Created Features

- hasIdentity
- Weekday
- Day
- Hour

Após separar as variáveis numéricas, categóricas, e as que serão usadas no PCA transformer (V1-V339), vamos a abordagem de `Pipeline` do Scikit-Learn para definir as transformações que vamos realizar.

As variavéis categóricas serão, primeiramente, transformadas em *string*, depois será feita a imputação de dados faltantes, usando a abordagem de mais frequente, ou seja, substitui os NaNs pela categoria mais frequente. Por fim, esses dados serão transformados em inteiros, usando o codificador `OrdinalEncoder`. Quando o modelo se deparar com uma categoria desconhecida, que não estava presente nos dados de treino, essa valor será codificado como -1.

Para as variáveis categórias usaremos uma abordagem de tratamento de valores faltantes similar, substituindo NaNs pela média. Após imputar os dados, usaremos o `StandarScaler` para uniformizar a distribuição dos dados.

As variáves V1 até V339, que foram submetidas à análise de componentes principais, vão passar pelo mesmo processo das features numéricas. Ao final do processo da padronização das escalas, será aplicada a redução de dimensionalidade, através da criação dos componentes principais.

Após definir os transformadores, vamos aplicar a `Pipeline` ao conjunto de treino e teste de forma distinta. Para o conjunto de treino, usamos o método `fit_transform()` que aprende a escala dos dados e cardinalidade das categorias, além de aplicar as transformações. Já o método `transform()` apenas vai aplicar no conjunto de teste as transformações necessárias aprendidas usando o conjunto de treino.

Ao realizar esse processo de forma separada, garantimos que as transformações não são influenciadas por informações dos dados teste, que devem ser totalmente desconhecidos para nosso modelo. Se houver influencia dos dados de teste nos dados de treino, estaremos diante de uma problema chamado **data leakage** ou vazamento de dados.

Antes das transformações nossos dados eram DataFrames do `pandas`. No entanto, o Scikit-Learn utiliza principalmente arrays para construir e treinar os modelos, portanto, nossos dados transformados de treino e teste agora são do tipo `ndarray`. Podemos excluir os antigos DataFrames de treino e teste para liberar espaço em memória.

## Seleção de Modelos

O processo de seleção de modelos realizado teve como objetivo comparar diferentes abordagens para prever a variável de interesse em um contexto de dados desbalanceados, utilizando a métrica ROC AUC como critério de avaliação.

O primeiro grupo de modelos avaliados foi composto por algoritmos simples, como DecisionTree, e algoritmos de ensemble, como é o caso do LightGBM (boosting), BaggingClassifier e Random Forest. Podemos observar que os algoritmos de boosting e Random Forest foram os que tiveram melhor performance, considerando a métrica escolhida.

Em seguida vamos testar técnicas específicas para dados desbalanceados, como undersampling e oversampling (SMOTE). OS modelos UNDER_RF e RUSB aplicam técnicas de undersampling, com a diferença que o primeiro é um algoritmo de bagging e o segundo de boosting. Já o SMOTE_LGBM é uma implementação de um algoritmo de boosting, utilizando uma técnica de oversampling, chamada SMOTE.

![Model Selection](/images/models.png)

O SMOTE_LGBM apresentou um comportamento muito atípico indicando sinais de overfitting, mas o LightGBM continua sendo um algoritmo com boa performance, mesmo sem aplicar técnicas de reamostragem. As implementações de RandomForest se mostraram bastante competitivas e as duas tiveram performances similares.

A diferença entre o UNDER_RF e RF não está na métrica de ROC AUC, mas sim em outras métricas importantes, que vamos discutir mais para frente. É importante considerar que, ao reduzir o número de exemplos, aplicando o undersampling o processo de treinamento é muito mais rápido, pois lida com um volume menor de exemplos. Em contrapartida, é preciso estar atento para perda de informações e o risco de underfitting.

## Treinamento

Considerando os testes realizados durante a seleção de modelos, vamos treinar diferentes modelos baseados em implementações do algoritmo `RandomForestClassifier`. O objetivo é encontrar a abordagem que retorna as melhores métricas, considerando nosso objetivos. Para isso, olharemos principalmente para Precisão, Recall e F2-Score.

O F2-Score é uma métrica de avaliação derivada do F1-Score, no entanto, atribui o dobro de peso ao recall em relação à precisão, diferente da métrica mais conhecida que busca o equilíbrio entre os dois. Isso significa que, ao calcular o F2-Score, erros que aumentam os falsos negativos (causando queda no recall) são penalizados de forma mais severa do que erros que afetam a precisão. Consequentemente, essa métrica é especialmente útil quando é mais importante identificar corretamente a maior quantidade possível de exemplos positivos (minimizando falsos negativos), mesmo que isso possa levar a alguns falsos positivos adicionais.

### Under Bagging

O método **Under Bagging** (Random Undersampler + Random Forest) consiste em aplicar uma estratégia de subamostragem aleatória para reduzir o tamanho da classe majoritária, seguido de um procedimento de agrupamento (bagging). A ideia é selecionar aleatoriamente um subconjunto dos dados da classe majoritária para balancear a distribuição, treinando vários modelos de Random Forest em diferentes subconjuntos. Ao fazer isso, o modelo passa a “enxergar” menos amostras da classe majoritária, o que tende a diminuir a ocorrência de falsos negativos e aumentar o foco na classe minoritária.

Considerando o trade-off entre precisão e recall, o Under Bagging pode melhorar o recall em cenários de alta desproporção entre classes,em função da queda da precisão, pois o modelo pode se tornar mais suscetível a classificar instâncias como pertencentes à classe minoritária.

### SMOTE Bagging

O **SMOTE Bagging** (SMOTE + Random Forest) utiliza a técnica de oversampling sintético (SMOTE) para gerar novas instâncias da classe minoritária. Nessa abordagem, em vez de reduzir a classe majoritária, cria-se artificialmente mais dados da classe minoritária, o que mitiga a perda de informação que ocorre no undersampling. O SMOTE gera exemplos adicionais interpolando pontos próximos no espaço de atributos, o que pode ajudar o modelo a aprender melhor as fronteiras da classe minoritária. Entretanto, se os dados sintéticos não forem bem representativos, o modelo pode “sobreajustar” padrões inexistentes.

A adição de exemplos sintéticos costuma melhorar o recall (reduzir falsos negativos), mas pode reduzir a precisão se o modelo acabar gerando mais falsos positivos.

### Adjusted Weights Random Forest

Outra abordagem é manipular o peso das classes de forma que a classe minoritária receba maior importância durante o treinamento. Isso faz com que os erros na classe minoritária “custem” mais, levando o modelo a se concentrar em diminuir os falsos negativos (aumentar o recall). Por outro lado, se os pesos da classe minoritária forem muito altos, o modelo pode superestimar essa classe e classificar incorretamente muitos exemplos da classe majoritária, diminuindo assim a precisão. O ajuste fino desses pesos é fundamental para encontrar um equilíbrio adequado entre precisão e recall, permitindo que o Random Forest lide melhor com o problema de desbalanceamento sem descartar dados ou criar instâncias sintéticas.

### Vanilla Random Forest

```markdown
ROC AUC: 0.8839
```

### Trade-off Precisão/Recall

O ajuste fino dos pesos das classes e da proporção de undersampling envolve testar diferentes configurações que priorizem ora a classe minoritária, ora a classe majoritária, a fim de equilibrar precisão e recall.

No caso dos pesos, é possível ajustar parâmetros como “class_weight” (no Random Forest) de modo que erros na classe minoritária tenham um custo mais elevado, enquanto na subamostragem (undersampling), varia-se a proporção de instâncias removidas da classe majoritária. O objetivo é encontrar um ponto ótimo em que o modelo consiga reduzir erros críticos (por exemplo, falsos negativos) sem penalizar demais a precisão, permitindo melhor adaptação a diferentes graus de desbalanceamento sem necessidade de descartar muitos dados ou criar instâncias sintéticas.

Para analisar o comportamente desses parâmetros vamos utilizar duas métricas: **F2-Score e Average Precision**

A métrica Average Precision (AP) mede a área sob a curva de Precisão vs. Recall ao longo de diferentes limiares de probabilidade. Em outras palavras, imagine que você gera vários valores de threshold para decidir quando um modelo classifica algo como positivo ou negativo, calculando a precisão e o recall em cada um desses valores. Em seguida, você plota os pontos em um gráfico com o Recall no eixo x e a Precisão no eixo y. A Average Precision é a área sob essa curva, ou seja, quanto maior ela for, melhor será o equilíbrio entre precisão e recall em todos os limiares.

### Otimização de Hiperparâmetros

Com base nas simulações realizadas observando que os parâmetros que obtiveram o melhor F2-Score foram uma proporção de 33% na redução da classe majoritária e o balanceamento da classes no treinamento do Random Forest. Usando essas informações vamos usar o GridSearch para otimizar os hiperparâmetros do modelo, usando validação cruzada. Com isso, teremos toda configuração necessária para treinar nosso modelo final.

## Modelo Final

A avaliação final do modelo foi feita utilizando um script de treinamento `train.py` que utiliza todos os dados de treino, ao invés da amostra usada no notebook para fins de experimentação.

```markdown
ROC AUC:    0.9162
Precision:  0.3149
Recall:     0.7055
F2 Score:   0.5653

Model training completed in 1199.94 seconds (20 minutes)
```
