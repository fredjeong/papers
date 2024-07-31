# Proximal Policy Optimization Algorithms

# 0. About this paper

- OpenAI에서 2017년 발표한 논문
- John Schulman의 policy optimisation 알고리즘
- 기존의 TRPO가 너무 복잡한 반면, 쉬운 구현과 상징성의 이유로 아직까지 가장 유명한 알고리즘으로 꼽힌다. TRPO를 발전시킨 것
- 데이터를 environment와 상호작용하면서 가져오고, SGD를 이용해 surrogate 목적함수를 업데이트
- https://ropiens.tistory.com/85
- [https://velog.io/@rockgoat2/Reinforcement-Learning-PPO-알고리즘-리뷰](https://velog.io/@rockgoat2/Reinforcement-Learning-PPO-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EB%A6%AC%EB%B7%B0)

# 1. Introduction
Trust region policy optimisation (TRPO) was suggested to address the issues with prior policy gradient methods [[1](https://proceedings.mlr.press/v37/schulman15.pdf)]. However, despite its efficiency and robustness, the method is relatively complicated compared to other methods and is not compatible with architectures that include noise (such as dropout) or parameter sharing (between the policy and value function).

The authors introduce proximal policy optimisation that shows as much efficiency and strong performance as TRPO only using first-order optimisation. The key idea in this paper is using clipped probability ratios that forms a lower bound of the performance.

# 2. Background: Policy Optimisation

## 2.1 Policy Gradient Methods

Policy gradient methods work by computing an estimator of the policy gradient and plugging it into a stochastic gradietn ascent algorithm. The estimator here is written as:

$$
\begin{align}
\nabla_\theta J(\theta)=\hat {\mathbb E}_t \left[ \nabla _\theta \log \pi _\theta(a_t\mid s_t)\hat A_t \right]
\end{align}
$$

where $\pi_\theta$ is the policy and $\hat A_t$ is an estimator of the advantage function at timestep $t$.

Therefore, we update the policy parameters by differentiating the loss function

$$
\begin{align}
L(\theta)= \hat{\mathbb E}_t \left[ \log \pi _\theta(a_t\mid s_t)\hat A_t \right].
\end{align}
$$

## 2.2 Trust Region Methods

In TRPO, an objective function is expressed as a surrogate function and is maximised subject to a constraint on the size of the policy update. That is, we define the problem as

$$
\begin{gather}
\max_\theta\hat{\mathbb E}_t\left[ \frac{\pi _\theta(a_t\mid s_t)}{\pi _{\theta _{old}}(a_t\mid s_t)} \hat A_t \right]
\\
\text{subject to }\hat{\mathbb E}_t\left[ \text{KL}[\pi _{\theta _{old}}(\cdot\mid s_t),\pi _\theta(\cdot\mid s_t)] \right]\leq\delta
\end{gather}
$$

where $\theta_{old}$ is the policy parameters before the update. This problem is approximately solved using the conjugate gradient algorithm, by linearly approximating the objective function and quadratically approximating the constraint.

Originally, the theory on which TRPO is based uses a penalty term instead of a constraint. That is, solving an unconstrained optimisation problem

$$
\max_\theta \hat{\mathbb E}_t \left[ \frac{ \pi _\theta(a_t\mid s_t)}{\pi _{\theta _{old}}(a_t\mid s_t)}\hat A_t - \beta\text{KL}[\pi _{\theta _{old}} (\cdot\mid s_t),\pi _\theta(\cdot\mid s_t)] \right]
$$

for some coefficient $\beta$. However, it is hard to choose the fixed value of $\beta$ as the characteristics change over the course of learning. Therefore, in this paper the authors add additional modificiations.

# 3. Clipped Surrogate Objective
Let $r_t$ denote the probability ratio $\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$ so we have $r(\theta_{old})=1$. In TRPO we maximise a surrogate objective function


$$
L^{CPI}(\theta)=\hat{\mathbb E}_t\left[ \frac{\pi _\theta(a_t\mid s_t)}{\pi _{\theta _{old}}(a_t\mid s_t)}\hat A_t \right]=\hat{\mathbb E}_t\left[ r_t(\theta)\hat A_t \right]
$$

Here, CPI stands for conservative policy iteration. Maximising $L^{CPI}$ will excessively update the policy without a constraint. Thus, it is required that we modify the objective in such a way that penalises changes to the policy that move $r_t(\theta)$ away from 1. That is, we maitain the "distance" between $\pi_\theta$ and $\pi_{\theta_{old}}$ by writing:

$$
L^{CLIP}(\theta)=\hat {\mathbb E}_t\left[ \min\left(r_t(\theta)\hat A_t, \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t \right)\right].
$$

By doing so, we remove the incentive for moving $r_t$ outside of the interval $[1-\epsilon,1+\epsilon]$. 
Initially, when $r_t(\theta)=1$, we have $L^{CPI}=L^{CLIP}$. As $\theta$ moves away from $\theta_{old}$ they become different. In the figure below we can see that the probability ratio $r$ is clipped at $1-\epsilon$ or $1+\epsilon$. This makes $L^{CLIP}$ as the lower bound of $L^{CPI}$ with **penalty for having too large of a policy update.**

![스크린샷 2024-07-31 15 31 13](https://github.com/user-attachments/assets/f9e76d52-67c0-4d5d-baf1-c9ad1d50fbee)

# 4. Adaptive KL Penalty Coefficient

Another approach is to use a penalty on KL divergence, and to adapt the penalty coefficient. That is, in the equation suggested in TRPO we change the value of $\beta$ adaptively.

$$
L^{KLPEN} \hat{\mathbb E}_t \left[ \frac{ \pi _\theta(a_t\mid s_t)}{\pi _{\theta _{old}}(a_t\mid s_t)}\hat A_t - \beta\text{KL}[\pi _{\theta _{old}} (\cdot\mid s_t),\pi _\theta(\cdot\mid s_t)] \right]
$$

However, this approach did not perform as well as the clipping method.

# 5. Algorithm

We use a neural network architecture that shares parameters between the policy and value function. For this, we must use a loss function that combines the policy surrogate and a value function error term. Adding an entropy bonus as in [[2](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)], we have

$$
\begin{align}
L_t^{CLIP+VF+S}(\theta)=\hat{\mathbb E}_t\left[ L _t^{CLIP}(\theta)-c _1 L _t^{VF}(\theta)+c_2 S[\pi _\theta](s _t) \right]
\end{align}
$$

where $c_1, c_2$ are coefficients, $S$ is an entropy bonus, and $L_t^{VF}$ is a squared-error loss $(V_\theta(s_t)-V_t^{targ})^2$

The algorithm is below. Each iteration, each of $N$ (parallel) actors collect $T$ timesteps of data (much less than the length of the episode). For this purpose, we use the concept of GAE (generalised advantage estimation) instead of $n$-step TD [[3](https://arxiv.org/pdf/1506.02438)]. We implement an advnatage estimator that does not look beyond timestep $T$, suggested by [[4](https://arxiv.org/pdf/1602.01783)]:

$$
\begin{align}
\hat A_t = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{T-t+1}r_{T-1}+\gamma^{T-t}V(s_T)
\end{align}
$$

where $t$ specifies the time index in [0,T] within a given length-$T$ trajectory segment. Using this, we find the exponential moving average of the infinite-step TD error as

$$
\begin{align}
\hat A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}
\end{align}
$$

where $\delta_t = r_t+\gamma V(s_{t+1} - V(s_t)}.

Then we construct the surrogate loss on these $NT$ timesteps of data, and optimise it with minibatch SGD or Adam, for $K$ epochs.

![스크린샷 2024-07-31 15 53 56](https://github.com/user-attachments/assets/86d17786-35b6-446f-b61d-b940233cd070)

# 6. Experiments

## 6.1 Comparison of Surrogate Objectives

## 6.2 Comparison to Other Algorithms in the Continuous Domain

## 6.3 Showcase in the Continuous Domain: Humanoid Running and Steering

## 6.4 Comparison to Other Algorithms on the Atari Domain

# 7. Conclusion

# 8. Acknowledgements
