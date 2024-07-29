# Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy

# 1. Introduction

It is challenging for analysts to consider all relevant factors in a complex and dynamic stock market.

A traditional approach:

1. The expected stock return and the covariance matrix of stock prices are computed. 
2. The best portfolio allocation strategy can be obtained by either maximising the return for a given risk ratio or minimising the risk for a pre-specified return.

Another approach is to model a Markov Decision Process and use dynamic programming to derive the optimal strategy. However, the scalability is limited due to the large state spaces when dealing with the stock market.

First, we build an environment and define action space, state space, and reward function. Second, we train the three algorithms that take actions in the environment. Third, we ensemble the three agents together using the Sharpe ratio that measures the risk-adjusted return.

# 2. Related Works

Recent applications of deep reinforcement learning in financial markets consider discrete or continuous state and action spaces, and employ one of these learning approaches: critic-only approach, actor-only approach, or actor-critic approach.

Using continuous action space provides better control capabilities than using discrete action space.

- Critic-only learning approach: solves a discrete action space problem using, for example, Deep Q-learning (DQN) and its improvements, and trains an agent on a single stock or asset. The idea is to use a Q-value function to learn the optimal action-selection policy that maximises the expected future reward given the current state. Instead of calculating a state-action value table, DQN minimises the error between estimated Q-value and target Q-value over a transition, and uses a neural network to perform function approximation.
    
    However, 
    
- Actor-only learning approach
- Actor-critic learning approach

# 3. Problem Description

## 3.1 MDP Model for Stock Trading

## 3.2 Incorporating Stock Trading Constraints

- Market liquidity: The orders can be rapidly executed at the close price and we assume that the stock market will not be affected by our reinforcement trading agent.
    
    → 시장이 물량을 받아줄 수 있고, 내 거래가 시장에 영향을 미치지 않는다.
    
- Nonnegative balance $b\geq0$: Let $\boldsymbol p_t^B=[p_t^i:i\in\mathcal B]$ and $\boldsymbol k_t^B=[k_t^i:i\in\mathcal B]$ be the vectors of price and number of buying shares for the stocks in the buying set. We can similarly define $\boldsymbol p_t^S$ and $\boldsymbol k_t^S$ for selling stocks, and $\boldsymbol p_t^H$ and $\boldsymbol k_t^H$ for the holding stocks. The cnostraint for non-negative balance can be expressed as
    
    $$
    b_{t+1}=b_t+(\boldsymbol p_t^S)^T\boldsymbol k_t^S - (\boldsymbol p_t^B)^T\boldsymbol k_t^B\geq0
    $$
    
- Transaction cost: There are many types of transaction costs such as exchange fees, execution fees, and SEC fees. We assume our transaction costs to be 0.1% of the value of each trade (either buy or sell):
    
    $$
    c_t= \boldsymbol p^T \boldsymbol k_t\times0.1\%
    $$
    
- Risk-aversion for market crash: To control the risk in a worst-case scenario like 2008 global financial crisis, we employ the financial turbulence index $turbulence_t$ that measures extreme asset price movements:
    
    $$
    turbulence_t=(\boldsymbol y_t - \boldsymbol \mu)\Sigma^{-1}(\boldsymbol y_t-\boldsymbol \mu)^T\in\mathbb R
    $$
    
    where $\boldsymbol y_t\in\mathbb R^D$ denotes the stock returns for current period $t$, $\boldsymbol \mu\in\mathbb R^D$ denotes the average of historical returns, and $\Sigma\in\mathbb R^{D\times D}$ denotes the covariance of historical returns. When $turbulence_t$ is higher than a threshold, which indicates extreme market conditions, we simply halt buying and the trading agent sells all shares. We resume trading once the turbulence index returns under the threshold.
    

## 3.3 Return Maximisation as Trading Goal

We define our reward function as the change of the portfolio value when action $a$ is taken at state $s$ and arriving at new state $s'$. 

The goal is to design a trading strategy that maximises the change of the portfolio value:

$$
r(s_t,a_t,s_{t+1})=(b_{t+1}+\boldsymbol p_{t+1}^T\boldsymbol h_{t+1})-(b_t+\boldsymbol p_t^T\boldsymbol h_t)-c_t
$$

Here, the first and the second term on the right hand side denote the portfolio value at $t+1$ and $t$, respectively.

To further decompose the return, we define the transition of the shares $\boldsymbol h_t$ as

$$
\boldsymbol h_{t+1}=\boldsymbol h_t-\boldsymbol k_t^S+\boldsymbol k_t^B
$$

Then we can rewritten the reward as

$$
r(s_t,a_t,s_{t+1})=r_H-r_S+r_B-c_t
$$

where

$$
\begin{gather}
r_H=(\boldsymbol p_{t+1}^H-\boldsymbol p_T^H)^T\boldsymbol h_t^H
\\
r_S=(\boldsymbol p_{t+1}^s-\boldsymbol p_t^S)^T\boldsymbol h_t^S
\\
r_B=(\boldsymbol p_{t+1}^B - \boldsymbol p_t^B)^T\boldsymbol h_t^B
\end{gather}
$$

Thus, we need to maximise the positive change of the portfolio value by buying and holding the stocks whose price will increase at next time step and minimise the negative change of the portfolio value by selling the stocks whose price will decrease at next time step.

When the turbulence index goes above a pre-specified threshold, we sell all we have. Thus, in this case for $r_S$ we have

$$
r_S=(\boldsymbol p_{t+1}-\boldsymbol p_t)^T\boldsymbol k_t
$$

which indicates that we want to minimise the negative change of the portfolio value by selling all held stocks.

We initialise the model as follows. We set $p_0$ to be the stock prices at time 0 and $b_0$ to be the amount of initial fund. In the first stage $h$ and $Q_\pi(s,a)$ are 0, and $\pi(s)$ is uniformly distributed among all actions for each state. Then we update $Q_\pi(s_t,a_t)$ through interacting with the stock market environment. 

The optimal strategy is given by the Bellman equations, such that the expected reward of taking action $a_t$ at state $s_t$ is the expectation of the summation of the direct reward $r(s_t,a_t,s_{t+1})$ and the future reward in the next state $s_{t+1}$.

Let the future rewards be discounted by a factor of $0<\gamma<1$ for convergence purpose, and we have

$$
Q_\pi(s_t,a_t)=\mathbb E_{s_{t+1}}[r(s_t,a_t,s_{t+1})+\gamma \mathbb E_{a_{t+1}\sim\pi(s_{t+1})}[Q_\pi(s_{t+1},a_{t+1})]]
$$

The goal is to design a trading strategy that maximises the positive cumulative change of the portfolio value $r(s_t,a_t,s_{t+1})$ in the dynamic environment, and we employ the deep reinforcement learning method to solve this problem.

# 4. Stock Market Environment

Before training a deep reinforcement trading agent, we carefully build the environment to simulate real world trading which allows the agent to perform interaction and learning. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. Our trading agent needs to obtain such information through the environment, and take actions defined in the previous section. We employ OpenAI gym to implement our environment and train the agent

## 4.1 Environment for Multiple Stocks

We use a continuous action space to model the trading of multiple stocks. We assume that our portfolio has 30 stocks in total.

$\{ buy, hold, sell\}$

### 4.1.1 State space

We use a 181-dimensional vector that consists of seven parts of information to represent the state space of multiple stocks trading environment: $[b_t, \boldsymbol p_t, \boldsymbol h_t, \boldsymbol M_t, \boldsymbol R_t, \boldsymbol C_t, \boldsymbol X_t]$:

- $b_t\in\mathbb R_+$: available balance at current time step $t$
- $\boldsymbol p_t\in\mathbb R_+^{30}$: adjusted close price of each stock
- $\boldsymbol h_t\in\mathbb Z_+^{30}$: shares owned of each stock
- $\boldsymbol M_t\in\mathbb R^{30}$: Moving Average Convergence Divergence (MACD) calculated using close price. MACD is one of the most commonly used momentum indicator that identifies moving averages.
- $\boldsymbol R_t\in\mathbb R_+^{30}$: Relative Strength Index (RSI) calculated using close price. RSI quantifies the extent of recent price changes. If price moves around the support line, it indicates the stock is oversold, and we can perform the buy action. If price moves around the resistance, it indicates the stock is overbought, and we can perform the selling action.
- $\boldsymbol C_t\in\mathbb R_+^{30}$: Commodity Channel Index (CCI) is calculated using high, low and close price. CCI compares current price to average price over a time window to indicate a buying or selling action.
- $\boldsymbol X_t\in\mathbb R^{30}$: Average Directional Index (ADX) is calculated using high, low and close price. ADX identifies trend strength by quantifying the amount of price movement.

### 4.1.2 Action space

For a single stock, the action space is defined as $\{-k,\cdots,-1,0,1,\cdots,k\}$, where $k$ and $-k$ present the maximum number of shares we can buy and sell, and $k\leq h_{max}$ while $h_{max}$ is a predefined parameter that sets as the maximum amount of shares for each buying action. 

Therefore, the size of the entire action space is $(2k+1)^{30}$. The action space is then normalised to $[-1,1]$, since the RL algorithms A2C and PPO define the policy directly on a Gaussian distribution, which needs to be normalised and symmetric.

→ 여기서 normalisation을 안시키면 discrete action space가 되는거고, 그때 DQN을 쓰면 되는거구나!

## 4.2 Memory Management

![스크린샷 2024-06-01 18.26.43.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e32255d7-f1d1-45eb-81e3-06e46c789c7a/a34afa2f-5da5-44fe-beea-873daaf203e8/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-06-01_18.26.43.png)

To tackle the problem of memory requirements, we employ a load-on-demand technique for efficient use of memory. The load-on-demand technique does not store all results in memory. Rather, it generates them on demand. The memory is only used when the result is requested, hence the memory usage is reduced.

# 5. Trading Agent Based on Deep Reinforcement Learning

## 5.1 Advantage actor critic (A2C)

A2C is a typical actor-critic algorithm, which is introduced to improve the policy gradient updates. It utilises an advantage function to reduce the variance of the policy gradient.  The critic network estimates not just the value function, but also the advantage function. Thus, the evaluation of an action depends on how good the action is AND how much better it can be. We can therefore reduce the high variance of the policy network and makes the model more robust.

It uses copies of the same agent to update gradients with different data samples. In each iteration, after all agents finish calculating their gradients, A2C uses a coordinator to pass the average gradients over all the agents to a global network. The global network can update the actor and the critic network. The presence of a global network increases the diversity of training data. 

The objective function for A2C is:

$$
\nabla J_\theta(\theta)=\mathbb E\left[ \sum_{t=1}^T \nabla_\theta\log \pi_\theta(a_t|s_t)A(s_t,a_t) \right]
$$

where $\pi_\theta(a_t|s_t)$ is the policy network, $A(s_t,a_t)$ is the advantage function that can be written as

$$
A(s_t,a_t)=Q(s_t,a_t)-V(s_t)
$$

or

$$
A(s_t,a_t)=r(s_t,a_t,s_{t+1})+\gamma V(s_{t+1})-V(s_t)
$$

## 5.2 Deep Deterministic Policy Gradient (DDPG)

DDPG is used to encourage maximum investment return. It combines the frameworks of both Q learning and policy gradient and uses neural networks as function approximators. 

In contrast with DQN that learns indirectly through Q-values tables and suffers the curse of dimensionality problem, DDPG learns directly from the observations through policy gradient. 

It is proposed to deterministically map states to actions to better fit the continuous action space environment. 

At each time step, the DDPG agent performs an action $a_t$ at $s_t$, receives a reward $r_t$ and arrives at $s_{t+1}$. The transitions $(s_t,a_t,s_{t+1}, r_t)$ are stored in the replay buffer $R$. A batch of $N$ transitions are drawn from $R$ and the Q-value $y_i$, $i=1,\cdots,N$, is updated as

$$
y_i = r_i + \gamma Q'(s_{i+1},\mu'(s_{i+1}|\theta^{\mu'},\theta^{Q'}))
$$

The critic network is then updated by minimising the loss function $L(\theta^Q)$ which is the expected difference between outputs of the target critic network $Q'$ and the critic network $Q$, i.e. 

$$
L(\theta^Q)=\mathbb E_{s_t,a_t,r_t,s_{t+1}\sim \text{buffer}}[(y_i-Q(s_t,a_t|\theta^Q))^2]
$$

DDPG is effective at handling continuous action space, and so it is appropriate for stock trading.

## 5.3 Proximal Policy Optimisation (PPO)

PPO is introduced to control the policy gradient update and ensure that the new policy will not be too different from the previous one.

PPO tries to simplify the objective of Trust Region policy Optimisation (TRPO) by introducing a clipping term to the objective function.

## 5.4 Ensemble Strategy

# 6. Performance Evaluations

## 6.1 Stock Data Preprocessing

We select the Dow Jones 30 constituent stocks (at 01/01/2016) as our trading stock pool. 

Our backtestings use historical daily data from 01/01/2009 to 05/08/2020 for performance evaluation.

The stock data can be downloaded from the Compustat database through the Wharton Research Data Services. 

Our dataset consists of two periods: in-sample period and out-of-sample period.

- In-sample period contains data for training and validation stages.
- Out-of-sample period contains data for trading stage.

In the training stage (01/01/2009 - 30/09/2015), we train three agents using PPO, A2C, and DDPG, respectively. 

In the validation stage (01/10/2015 - 31/12/2015), we validate the three agents by Sharpe ratio, and adjust key parameters, such as learning rate, number of episodes, etc.

In the trading stage (01/01/2016 - 08/05/2020), we evaluate the profitability of each algorithm.

To better exploit the trading data, we continue training our agent while in the trading stage, since this will help the agent to better adapt to the market dynamics.

## 6.2 Performance Comparisons

# 7. Conclusion
