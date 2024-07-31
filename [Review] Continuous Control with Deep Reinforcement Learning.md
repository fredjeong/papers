# Continuous Control with Deep Reinforcement Learning
- Author: Timothy P. Lillicrap , Jonathan J. Hunt , Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra
- Journal: ICLR
- Year: 2016
- Link: [[pdf](https://arxiv.org/pdf/1509.02971)]

## 1. Introduction

DQN [[1](https://arxiv.org/pdf/1312.5602)] can only handle discrete and low-dimensional action spaces while many real-life tasks have continuous and high dimensional action spaces. In these cases, DQN cannot be directly applied to continuous domains as it requires an iterative optimisation process at every step.

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

In continuous action spaces finding the greedy policy requires an optimisation of $a_t$ at every tmiestep, which is too slow to be practical with large and unconstrained function approximators and nontrivial action spaces. Instead, we use an actor-critic approach based on the DPG algorithm. 

We maintain a parameterised actor function $\mu(s\mid \theta^\mu) that specifies the current policy by mapping states to a specific action deterministically. The critic $Q(s,a)$ is then updated by applying the chain rule to the expected return from the start distribution $J$ with respect to the actor parameters:

$$
\begin{align}
\nabla_{\theta^\mu}J
&\approx \mathbb E\left[\nabla_{\theta^\mu} Q(s,a\mid \theta^Q)| _{s=s_t, a=\mu(s_t\mid\theta^\mu)}\right]
\\
&= \mathbb E\left[ \nabla_a Q(s,a\mid\theta^Q)| _{s = s_t,a = \mu(s_t)} \nabla _{\theta^\mu}\mu(s\mid\theta^\mu)| _{s=s_t} \right]
\end{align}
$$

which is the policy gradient (see [[2](https://proceedings.mlr.press/v32/silver14.pdf)] for proof).

There are a few challenges with using neural networks for reinforcement learning. First, most optimisation algorithms assume that the samples are independently and identically distributed. However, when the samples are generated from exploring sequentially in an environment this assumption no longer holds. DQN addressed this issue by using a replay buffer. We store the trajectories $(s_t,a_t,r_t,s_{t+1})$ and at each timestep the actor and critic are updated by smapling a minibatch uniformly from the buffer. Because the DDPG is an off-policy algorithm, the replay buffer can be large, allowing the algorithm to benefit from learning across a set of uncorrelated transitions.

Second, as in Q-learning, introducing non-linear function approximators means that convergence is no longer guaranteed since the network $Q(s,a\mid \theta^Q)$ being updated is also used in calculating the target value. Therefore, we create a copy of the actor and critic networks, $Q'(s,a\mid \theta^{Q'}$ and $\mu'(s\mid \theta^{\mu'}$ respectively and use them to calculate the target values. The weights of these target networks are then updated by having them track the learned networks, i.e. $\theta'\leftarrow \tau\theta+(1-\tau)\theta'$ with $0<\tau<1$. This way, the target values are constrained to change slowly and improves the stability of learning. 

Another problem is that when learning from low dimensional feaure vector observations, the differnet components of the observation may have different physical units (e.g. positions versus velocities). This makes it difficult for the network to learn effectively and makes it difficult to find hyperparameters which generalise across environments with different scales of state values. Although we acn manually scale the features so they are in similar ranges across environments and units, we can also use batch normalisation, a technique that normalises each dimension across the samples in a minibatch to have unit mean and variance. This way we can learn across different tasks with differing types of units without needing to manually ensure the units were within a set range.

Lastly, exploration can be a problem in continuous action spaces. By using off-policy algorithms we can treat the problem of exploration independently from the learning algorithm. We construct an exploration policy $\mu'$ by adding noise sampled from a noise process $\mathcal N$ to our actor policy

$$
\begin{align}
\mu'(s_t)=\mu(s_t\mid \theta_t^\mu)+\mathcal N
\end{align}
$$

![스크린샷 2024-07-31 19 01 53](https://github.com/user-attachments/assets/c61cee8d-7aae-4fe3-a406-f5bec8f471f5)

## 4. Results

## 5. Related Work

## 6. Conclusion
