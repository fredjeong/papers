# Cryptocurrency Portfolio Management with Deep Reinforcement Learning

# 0. Abstract

Portfolio management is the decision-making process of allocating an amount of fund into different financial investment products. Cryptocurrencies are electronic and decentralized alternatives to government-issued money, with Bitcoin as the best-known example of a cryptocurrency. This paper presents a model-less convolutional neural network with historic prices of a set of financial assets as its input, outputting portfolio weights of the set. The network is trained with 0.7 years’ price data from a cryptocurrency exchange. The training is done in a reinforcement manner, maximising the accumulative return, which is regarded as the reward function of the network. Back test trading experiments with trading period of 30 minutes is conducted in the same market, achieving 10-fold returns in 1.8 month’s periods. Some recently published portfolio selection strategies are also used to perform the same back tests, whose results are compared with the neural network. The network is not limited to cryptocurrency, but can be applied to any other financial markets.

# 1. Introduction

The authors apply a full machine learning approach to the general portfolio management problem, without assuming any prior knowledge of the financial markets or making any models, and completely letting the algorithms observe and learn from the market history.

Previous works that applied deep machine-learning techniques to financial market trading focused on predicting the price movements using historic market data. That is, the model takes as input a history price matrix and outputs a vector predicting the prices in the next period.

However, the model that the authors suggest does not predict the price of any specific financial product. Rather it directly outputs the market management actions, the portfolio vector.

This is primarily due to two reasons. First, if the model only outputs a price vector, humans should intervene and decide the number of shares of each position (buy and sell). Second, high accuracy in predicting price movement is usually difficult to achieve, while the ultimate goal of portfolio management is to make higher profit instead of higher price-prediction accuracy.

The authors set the action space as CONTINUOUS.

→ Although market actions can be discretised, it is considered a drawback as it discrete actions come with unknown risks. Also, discretisation scales badly, while market factors such as the number of total assets vary.

- Q-learning is limited to problems with discrete actions.
- critic-actor Deterministic Policy Gradient outputs continuous actions, training a Q-function estimator as the reward function, and a second neural network as the action function. However, training two neural networks is difficult and sometimes unstable.
- In this paper the authors employ a simple deterministic policy gradient using a direct reward function in the portfolio management problem, avoiding Q-function estimation

Two natures of cryptocurrencies, differentiate them from traditional financial assets, making their market the best test ground for our novel machine-learning portfolio management experiments. These natures are decentralisation and openness, and the former implies the latter. 

Without a central regulating party, anyone can participate in cryptocurrency trading with low
entrance requirements, and cryptocurrency exchanges flourish. One direct consequence is abundance of small-volumed currencies. Affecting the prices of these penny-markets will
require smaller amount of investment, compared to traditional markets. This will eventually allow trading machines learning and taking the advantage of the impacts by their own market actions. 

Openness also means the markets are more accessible. Most cryptocurrency exchanges have application programming interface for obtaining market data and carrying out trading actions, and most exchanges are open 24/7 without restricting the number of trades. These non-stop markets are ideal for machines to learn in the real world in shorter time-frames.

# 2. Problem Definition

## 2.1 Problem Setting

- $m$: number of assets selected to be traded
- $n$: trading periods
- Together $m$ and $n$ construct the global price matrix $G$
    
    ![스크린샷 2024-06-11 12.58.25.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e32255d7-f1d1-45eb-81e3-06e46c789c7a/20c0eee3-b3d9-483d-b3ae-a0b336eca8eb/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-06-11_12.58.25.png)
    
    where $x_{(i,t)}$ is the price of $i$-th asset at the beginning of the $t$-th trading period. 
    
    Each row of the matrix represents the price time-sequence of an asset.
    
    Especially, the first row is the riskless asset. For example, in this case the riskless asset is Bitcoin whose price is always 1, and all the other prices are the exchange rates against Bitcoin. The $t$-th column of the matrix is the price vector, denoted by $v_t$ of $t$-th trading period.
    

By element-wise dividing $\boldsymbol v_{t+1}$ by $\boldsymbol v_t$, we get price change vector of $t$-th trading period $\boldsymbol y_t$:

![스크린샷 2024-06-11 17.58.46.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e32255d7-f1d1-45eb-81e3-06e46c789c7a/4cb196d7-d98f-4376-a94e-be23d91b1bfb/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-06-11_17.58.46.png)

Suppose an agent is investing on the market, and his investment on a trading period $t$ is specified by a portfolio vector 

$$
\boldsymbol w_t=(w_{(t,1)},\cdots,w_{(t,i)},\cdots, w_{(t,m)})
$$

where $w_{(t,i)}$ represents the proportion of total capital invested in the $i$-th capital, and thus $\sum_iw_{(t,i)}=1$ for all $t$.

In a portfolio management problem, the initial portfolio vector $\boldsymbol w_0$ is chosen to be the first basis vector in the Euclidean space, that is $\boldsymbol w_0=(1,0,\cdots,0)$, indicating all the capital is in the riskless asset or in a fiat currency, before the first trading period. It is Bitcoin in our case. → 그러면 거래를 원화나 파운드화가 아니라 비트코인으로 하는거네? 비트코인을 기축통화처럼 썼다는 거 아냐?

Ignoring the transaction fee, the dot product of portfolio vector $\boldsymbol w_t$ in the current period $t$, and the price change vector $\boldsymbol y_t$ of the next, is the capital change rate $r_t$ (i.e. total capital in next period divided by that of this period) for the next trading period.

$$
r_t=\boldsymbol w_t\cdot \boldsymbol y_t
$$

If the commission fee is $C$ per Bitcoin, the total transaction fee in $t$-th trading period is then:

$$
\mu_t=C\sum_{i=1}^m|\boldsymbol w_{(t-1,i)}-\boldsymbol w_{(t,i)}|
$$

In our scenario, $C=0.0025$, the maximum commission rate at Poloniex. ← 내 문제 상황에 맞게 수정할 것

After $n$ trading periods the portfolio value, which is equal to initial portfolio value plus the total return, $\alpha_n$, becomes:

$$
\begin{align}
\alpha_n &= \prod_{t=0}^n r_t(1-\mu_t)
\\
&= \prod_{t=0}^n \boldsymbol w_t\cdot \boldsymbol y_t(1- C\sum_{i=1}^m| w_{(t-1,i)} - w_{(t,i)}|)
\end{align}
$$

where the unit of portfolio value is chosen such that $\alpha_0=1$.

At the beginning of each trading period $t$, the agent obtains $m$ sequences of history prices, and based on them, makes the investment decision $\boldsymbol w_t$.

This process will repeat until the last trading period. 

The purpose of our algorithmic agent is to generate, in this process, a sequence of portfolio vector $\{\boldsymbol w_1, \boldsymbol w_2, \cdots, \boldsymbol w_n\}$ in order to maximise the accumulative capital.

**Assumptions**

- Market liquidity: Each trade can be finished immediately at the last price when the orders are put.
- Capital impact: The capital invested by the algorithm is so insignificant that is has no influence on the market.

# 3. Data

The price data obtained from Poloniex is one year in time span and the trading period is half an hour. → 30분에 한 번씩 거래 진행

All data is constructed into a global price matrix $G$.

The input of the CNN is an $m\times w$ price matrix in the $t$-th trading period $X_t$, of which each row is the price sequence of a coin during last $w$ trading periods, a trading window. In our experiment, $m=12$, $w=50$. → 그러면 뒤에 나올 가격을 알고 거래하는 게 아닌가? / 코인은 12개로 실험했음

## 3.1 Coin Selection

## 3.2 Data Preprocessing

- Normalisation: The absolute price values of the assets in the problem are not important for the agent to make any trading decisions, but only changes in price matter.
    
    → Input prices to the network are normalised, dividing the current price vector.
    
    For an input window of $w$ periods, the authors define a local normalised price matrix, or simply price matrix, feeding the neural network. The price matrix reads:
    
    ![스크린샷 2024-06-13 17.47.56.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e32255d7-f1d1-45eb-81e3-06e46c789c7a/92398220-2372-48d8-88ba-6076fd33b64b/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-06-13_17.47.56.png)
    
    To train the network, the authors defined the price change vector of the period $\boldsymbol y_t$ to define the reward function.
    
- Filling empty history data: The data before the existence of a coin is marked as NaN.
    
    As the input of the CNN must be real numbers, these NANs should be replaced. The simplest approach is to replace them with 1, indicating the price did not fluctuate before launching. However, during the training, it is meaningless to invest the nonexistent asset (but it’s not cheating because there is the riskless asset, Bitcoin, in the assets set). Moreover, this part
    of history may also be learnt by the CNN and such asset may be recognised as riskless. It is not expected that the algorithm to invest in coins which lacks a large part of history, because
    less training data means a higher probability of over-fitting. Therefore, a fake decreasing price series is filled with decay rate 0.01 in the blank history for each coin if necessary, in
    order to prevent the agent from investing that asset. Note that the decay rate can not be set bigger than 0.05 or the training process will be easily trapped in local minima.
    

## 3.3 Dividing Data into Three Sets

The global price matrix $G$ is divided into three parts: training, test, and cross-validation sets. 

The cross-validation set is used to tune hyperparameters, such as the number of neurons in the hidden fully-connected layer of the network. The ratio among these three sets is 0.7:0.15:0.15.

## 3.4 Perspective of Reinforcement Learning

The total capital change after each trading period is the reward.

The output portfolio vector $\boldsymbol w_t$ is the action and the history price matrix $X_t$ is used to represent the state of the market.

Therefore, the whole portfolio management process of $n$ trading periods can be represented as a state-action-reward-state trajectory $\tau=(X_1,\boldsymbol w_1, r_1, X_2,\boldsymbol w_2, r_2,\cdots, X_n,\boldsymbol w_n,r_n)$.

Note that the action $\boldsymbol w_t$ will not influence the state information in the next period $X_{t+1}$ due to our assumption. As the experiment method being back-test, which uses history data to mimic a real trading, cannot provide such influence.

# 4. Deterministic Policy Gradient

## 4.1 Portfolio Weight as Output

Traditional ways of using CNN is to predict the change in price, so the output is predicted price vector.

Common policy gradient networks output the probability of each action, limiting the action to discrete cases. 

The authors suggest a model that directly outputs the portfolio weight vector, whose element is the ratio of total capital. For example, if the first element of the vector is 0.2, the algorithm will keep 20% of the total capital in the first asset.

## 4.2 Reward Function

The reward function is defined as:

$$
R_0=\sqrt[n]{\prod_{t=1}^{n-1}\boldsymbol w\cdot\boldsymbol y_t \left(1 - C\sum_{i=1}^m |w_{(t,i)}-w_{(t+1,i)}|\right)}
$$

As the input matrix does not include portfolio vector $\boldsymbol w$ of the last period, adding the transaction cost term into the reward function will not be helpful but will slow down the training, so the authors ignore this term.

For computational efficiency, we take the logarithm and the final reward function, which is the average logarithmic return, is

$$
R=\frac{1}{n}\sum_{t=1}^{n+1}\log (\boldsymbol w_t\cdot \boldsymbol y_t)
$$

Each portfolio vector $\boldsymbol w_t$ satisfies $\sum_{i}w_{(t,i)}=1$, so we use softmax as the activation function in the output layer.

# 5. Network Training

# 6. Network Topology

## 6.1 Model Selection

## 6.2 CNN Topology

## 6.3 Fully Connected Network

# 7. Performance Evaluation

## 7.1 Results

## 7.2 The Expiration Problem

## 7.3 Dilemma Between Performance Evaluation and Hyperparameters tuning

# 8. Conclusion
