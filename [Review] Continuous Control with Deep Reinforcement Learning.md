# Continuous Control with Deep Reinforcement Learning
- Author: Timothy P. Lillicrap , Jonathan J. Hunt , Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra
- Journal: ICLR
- Year: 2016
- Link: [[pdf](https://arxiv.org/pdf/1509.02971)]

## 1. Introduction

DQN [[1](https://arxiv.org/pdf/1312.5602)]can only handle discrete and low-dimensional action spaces while many real-life tasks have continuous and high dimensional action spaces. In these cases, DQN cannot be directly applied to continuous domains as it requires an iterative optimisation process at every step.

Simply discretetising the action space has several limitations. First is the curse of dimensionality. That is, the number of actions increases exponentially with the number of degrees of freedom. Second, naive discretisation gives up information about the structure of the action domain.

We develop the deterministic policy gradient (DPG) algorithm [[2](https://proceedings.mlr.press/v32/silver14.pdf)] by combining it with DQN. Two ideas from DQN were used: (1) train the network off-policy with samples from a replay buffer to minimise correlations between samples and (2) the network is trained with a target Q network to give consistent targets during temporal difference backups. Moreover, batch normalisation [[3](https://arxiv.org/pdf/1502.03167)] is also used.

## 2. Background

Let $R_t=\sum _{i=t} ^T \gamma^{(i-t)}r(s_i,a_i)$ be the sum of discounted future reward with a discounting factor $\gamma\in[0,1]$. Our goal is to learn a policy which amximies the expected return from the start distribution $J=\mathbb E[R_0]$. We denote $\rho^\pi$ for the discounted state visitation distribution for a policy $\pi$.

The action-value function $Q$ is defined as:

$$
\begin{align}
Q^\pi(s_t,a_t) 
&= \mathbb E[R_t\mid s_t,a_t]
\\
&= \mathbb E[r(s_t,a_t)+\gamma \mathbb E[Q^\pi(s_{t+1},a_{t+1}]]
\end{align}
$$

For a deterministic target policy, we use $\mu$ and avoid the inner expectation as below:

$$
\begin{align}
Q^\mu(s_t,a_t) = \mathbb E[r(s_t,a_t)+\gamma Q^\mu(s_{t+1},\mu(s_{t+1}))]
\end{align}
$$

Note that **the expectation above depends only on the environment (state and reward),** meaning that we can learn $Q^\mu$ off-policy, using transitions which are generated from a different stochastic behaviour policy $\beta$.

We approximate the action-value function parameterised by $\theta^Q$ and optimise it by minimising the loss:

$$
\begin{align}
L(\theta^Q)=\mathbb E\left[ \left(Q(s_t,a_t\mid \theta^Q)-y_t\right)^2 \right]
\end{align}
$$

where $y_t=r(s_t,a_t)+\gamma Q(s_{t+1},\mu(s_{t+1})\mid \theta^Q)$.

The use of large, non-linear function approximators for learning value or action-value functions has been ignored since theoretical performance guarantees are impossible and practically learning tends to be unstable. However, we use a **replay buffer** and a **separate target network** for calculating $y_t$.


## 3. Algorithm

## 4. Results

## 5. Related Work

## 6. Conclusion
