# Deterministic Policy Gradient Algorithms

# 0. About this paper

- DeepMind에서 2016년에 발표
- 2014년 공개된 DPG(Deterministic Policy Gradient) 논문에서 DQN을 결합하여 발전시킨 알고리즘
- DQN에서 성공적이었던 부분들을 continuous action 영역으로 확장
    - 단순히 continuous action 영역을 discrete한 구간으로 잘게 나누는 것은 많아진 discrete action들로 인한 학습 성능 저해(curse of dimensionality)와 기존 continuous action이 가진 구조적 정보를 잃어버린다는 단점 존재
    - 따라서 continuous action 영역에서 정의된 deterministic policy를 이용하는 상황에서 DQN 알고리즘의 장점을 결합
- 기존 DPG 알고리즘에 딥러닝을 통한 non-linear function approximation, target network, replay buffer를 도입
- [https://ai-com.tistory.com/entry/RL-강화학습-알고리즘-4-DDPG](https://ai-com.tistory.com/entry/RL-%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-4-DDPG)

# 1. Introduction

Policy gradient는 continuous action space에서 동작하는 알고리즘임

기본적인 아이디어는 policy를 parametric probability distribution $\pi_\theta(a\mid s)=\mathbb P[a\mid s;\theta]$로 둬서 어떤 state $s$ 아래에서 매개변수화된 stochastic action $a$를 표현하는 것.

그래서 이러한 stochastic policy에서 표본을 sampling하여 policy를 업데이트하고 policy parameters $\theta$를 수정함 (in the direction of greater cumulative reward).

본 논문에서는 그 대신 deterministic policy $a=\mu_\theta(s)$를 상정함. 

Peters, 2010에서는 deterministic policy gradient는 존재하지 않거나 모형을 사용했을 시에만 얻을 수 있다고 했으나 본 논문에서는 deterministic policy gradient가 존재하며, 꽤 간단한 form으로 표현됨을 보임.

본 논문은 deterministic policy gradient가 stochastic policy gradient에서 variance가 0이 되는 특수한 상황이라는 것을 보임.

Stochastic policy gradient의 경우 state와 action 모두를 적분해야 하므로 더 많은 샘플이 필요함 (action space의 차원이 커질수록 이러한 경향은 심화됨).

state space와 action space를 충분히 탐험(explore)하기 위해서 본 논문은 off-policy learning algorithm을 소개함. 이 방식은 stochastic behaviour policy에 따라 action을 선택하지만 학습은 deterministic target policy에서 이루어짐.

Finally, there are many applications (for example in robotics) where a differentiable control policy is provided, but where there is no functionality to inject noise into the controller. In these cases, the stochastic policy gradient is inapplicable, whereas our methods may still be useful.

# 2. Background

## 2.1 Preliminaries

$t$ time steps 이후 state $s$에서 $s'$가 되었다고 하자. 

이 때, discounted state distribution은 다음과 같이 나타낼 수 있다.

$$
\rho^\pi(s')\coloneqq \int_{\mathcal S}\sum_{t=1}^\infty \gamma^{t-1}p_1(s)p(s\to s',t,\pi)~\mathrm{d}s
$$

이 때, objective function을 다음과 같이 쓸 수 있다.

$$
\begin{align}
J(\pi_\theta)
&=\int_{\mathcal S}\rho^\pi(s)\int_{\mathcal A}\pi_\theta(s,a)r(s,a)~\mathrm{d}a\mathrm{d}s
\\
&=\mathbb E_{s\sim \rho^\pi,a\sim\pi_\theta}[r(s,a)]
\end{align}
$$

## 2.2 Stochastic Policy Gradient Theorem

Stochastic policy gradient 알고리즘은 $\nabla_\theta J(\pi_\theta)$의 방향으로 policy update를 진행한다. 이는 Sutton et al., 1999에서 policy gradient theorem을 이용해 나타낸다.

$$
\begin{align}
\nabla_\theta J(\pi_\theta)
&= \int_{\mathcal S}\rho^\pi(s)\int_{\mathcal A}\nabla_\theta\pi_\theta(a\mid s)Q^\pi(s,a)~\mathrm{d}a\mathrm{d}s
\\
&= \mathbb E_{s\sim \rho^\pi, a\sim \pi_\theta}[\nabla_\theta\log \pi_\theta(a\mid s)Q^\pi(s,a)]
\end{align}
$$

여기서 우리는 비록 state distribution $\rho^\pi(s)$가 policy parameters $\theta$에 의해 결정되지만 policy gradient는 state distribution의 gradient에 의존하지 않음을 알 수 있다.

Policy gradient 알고리즘이 가지고 있는 이슈 중 하나는 어떻게 action-value function $Q^\pi(s,a)$를 추정할 것인지다. 이에 대해 가장 간단한 방법은 Williams, 1992가 제안한 REINFORCE algorithm의 변종으로서, sample return $r_t^\gamma$를 이용해 $Q^\pi(s_t,a_t)$의 값을 추정하는 것이다.

## 2.3 Stochastic Actor-Critic Algorithms

Actor-critic 아키텍쳐는 policy gradient theorem에 기반한다.

Actor는 stochastic gradient ascent 방식으로 위의 식 (4)에 기반해 stochastic policy $\pi_\theta(s)$의 policy parameter $\theta$를 업데이트한다. 여기서 $Q^\pi(s,a)$를 알지 못하기 때문에 $Q^w(s,a)$로 매개변수화시킨다. 

Critic은 temporal-difference learning 등의 알고리즘을 사용해 $Q^w(s,a)\approx Q^\pi(s,a)$가 되도록 추정한다.

하지만 이렇게 $Q^w$로 대체하여 사용하는 방식에는 bias가 존재할 수 있으나, $Q^w(s,a)=\nabla_\theta\log\pi_\theta(a\mid s)^Tw$이고 $w$가 mean squared error $\epsilon^2(w)=\mathbb E_{s\sim \rho^\pi,a\sim\pi_\theta}[(Q^w(s,a)-Q^\pi(s,a))^2]$를 최소화한다면 bias는 존재하지 않는다. 하지만 이 두 방식이 충족되었을 경우에는 critic을 사용하지 않는 것과 같고, REINFORCE 알고리즘과 별로 다를 게 없어진다. (Williams, 1992, Sutton et al., 2000).

## 2.4 Off-Policy Actor-Critic

# 3. Gradients of Deterministic Policies

## 3.1 Action-Value Gradients

## 3.2 Deterministic Policy Gradient Theorem

## 3.3 Limit of the Stochastic Policy Gradient

# 4. Deterministic Actor-Critic Algorithms

## 4.1 On-Policy Deterministic Actor-Critic

## 4.2 Off-Policy Deterministic Actor-Critic

## 4.3 Compatible Function Approximation

# 5. Experiments

## 5.1 Continuous Bandit

## 5.2 Continuous Reinforcement Learning

## 5.3 Octopus Arm

# 6. Discussion and Related Work

# 7. Conclusion
