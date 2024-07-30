# Human-level Control Through Deep Reinforcement Learning

- Author: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, Demis Hassabis
- Journal: Nature
- Year: 2015
- Link: [[pdf](https://daiwk.github.io/assets/dqn.pdf)]

---
In this paper the authors introduce a artificial agent called a deep Q-network. A notable difference that a DQN agent has is that it learns **directly from inputs** using end-to-end reinforcement learning. In the experiment, receiving only the pixels and the game scores as inputs, the agent outperformed all algorithms invented prior to 2015. 

The authors used the deep convolutional network, which uses hierarchical layers of toiled convolutional filters to mimic the effects of receptive fields and therefore are able to exploit the local spatial correlations present in images.

The tasks given are such that the agent **interacts with an environment** through a sequence of **observations, actions and rewards**, where the goal of the agent is to select such actions that maximise cumulative future reward. Mathematically, this process is using a deep convolutional neural network to approximate the optimal **action-value function:

$$
\begin{align}
Q^*(s,a) = \max_\pi \mathbb E\left[ r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots \mid s_t=s, a_t=a, \pi \right]
\end{align}
$$

where $r$ is the reward, $s_t$ is the state at time $t$, $a_t$ is the action chosen at time $t$, $\gamma$ is the discount factor, and $\pi=P(a\mid s)$ is the behavioural policy according to which the agent chooses actions.

It is known that reinforcement learning is unstable and convergence is not guaranteed when a non-linear function approximator, one of which is a neural netowrk, is used to represent the action-value function $Q$ [[1](https://proceedings.neurips.cc/paper_files/paper/1996/file/e00406144c1e7e35240afed70f34166a-Paper.pdf)], possibly due to the correlations in the sequence of observations, small updates to $Q$ significantly changing the policy and thereby changing the data distribution, and the correlations between $Q$ and the target values $r+\gamma \max_{a'}Q(s',a')$.

The authors addressed this problem by (1) suggesting the concept of **experience replay** that randomises over the data, removing correlations in the observation sequence and smoothing over changes in the data distribution and (2) using an iterative update that adjusts the action-values $Q$ towards target values that are only periodically updated, resulting in reduced correlations with the target.

Although there already exists other stable methods for training neural networks in the reinforcement learning setting, such as neural fitted Q-iteration, they are too inefficient to be used with large neural networks.

In a deep Q-network, the authors parameterise an approximate value function $Q(s,a;\theta_i)$ using the deep convolutional neural network, where $\theta_i$ are the parameters of the Q-network at iteration $i$. 

The agent's experiences at each time-step $t$ are stored as **trajectories** in a dataset $D_t=\{e_1,\cdots,e_t\}$ where $e_t=(s_t,a_t,r_t,s_{t+1})$. During learning, Q-learning updates are applied on samples of experience $(s,a,r,s')\sim U(D)$. These samples are drawn uniformly from $D$. The resulting update at iteration $i$ uses the following loss function:

$$
\begin{align}
L_i(\theta_i) = \mathbb E_{(s,a,r,s')\sim U(D)}\left[ \left( r+\gamma \max_{a'} Q(s',a';\theta_i^-) - Q(s,a;\theta_i) \right)^2 \right].
\end{align}
$$

Note that here the authors differentiated $\theta_i$ and $\theta_i^-$, where $\theta_i^-$ are the network parameters used to compute the target at iteration $i$. It is only updated with the Q-network parameters every $C$ steps and are held fixed between individual updates. 

## Experiments

The authors used the Atari 2600 platform to test the DQN agent. When compared with the best performing reinforcement learning methods, on 43 games out of 43 the DQN agent outperformed them. It is worth noting that the games in which DQN excels are varied in their nature, from side-crolling shooters to boxing games and three-dimensional car-racing games, although games demanding more temporally extended planning strategies are challenging for all existing agents including DQN.

## Training Algorithm for deep Q-networks**

![스크린샷 2024-07-29 21 15 43](https://github.com/user-attachments/assets/d9179419-3c50-44d2-939e-7ec81759d57f)

The agent selects and executes actions according to an $\epsilon$-greedy policy based on $Q$. The authors fixed the length of representation of histories produced by the function **$\phi$**. 

The algorithm modifies standard online Q-learning in two ways to make it suitable for training large neural networks without diverging. 

1. Experience replay is used. The authors store the agent's experiences at each time-step, $e_t=(s_t,a_t,r_t,s_{t+1}$, in a dataset $D_t=\{e_1,\cdots,e_t\}$, pooled over many episodes in to a replay memory. During the inner loop of the algorithm, Q-learning updates are applied to samples for experience, $(s,a,r,s')\sim U(D)$, drawn randomly from the pool of stored samples. By using experience replya the behaviour distribution is averaged over many of its previous states, smoothing out learning and avoiding oscillations or divergence in the parameters.

> This approach has several advantages over standard online Q-learning.
> 
> First, each step of experience is potentially used in many weight updates, which allows for greater data efficiency.
> 
> Second, learning directly from consecutive samples is inefficient, owing to the strong correlations between the samples; randomizing the samples breaks these correlations and therefore reduces the variance of the updates.
> 
> Third, when learning onpolicy the current parameters determine the next data sample that the parameters are trained on. For example, if the maximizing action isto move left then the training samples will be dominated by samples from the left-hand side; if the maximizing action then switches to the right then the training distribution will also switch. It is easy to see how unwanted feedback loops may arise and the parameters could get stuck in a poor local minimum, or even diverge catastrophically.

*When learning by experience delay, it is necessary to learn **off-policy**, because the current parameters are different to those used to generate the sample, which motivates the choice of Q-learning.

In practice, the authors stored the last $N$ experience tuples in the replay memory and samples uniformly at random from $D$ when performing updates.

2. A separate network for generated the targets $y_i$ in the Q-learning update. Every $C$ updates the authors clone the network $Q$ to obtain a target network $\hat Q$ and use $\hat Q$ for generating the Q-learning targets $y_i$ for the following $C$ updates to $Q$.

3. Clipping the error term from the update $r+\gamma \max_{a'}Q(s',a';\theta_i^-)-Q(s,a;\theta_i)$ to be between -1 and 1. 
