# Proximal Policy Optimization Algorithms

# 0. About this paper

- OpenAI에서 2017년 발표한 논문
- John Schulman의 policy optimisation 알고리즘
- 기존의 TRPO가 너무 복잡한 반면, 쉬운 구현과 상징성의 이유로 아직까지 가장 유명한 알고리즘으로 꼽힌다. TRPO를 발전시킨 것
- 데이터를 environment와 상호작용하면서 가져오고, SGD를 이용해 surrogate 목적함수를 업데이트
- https://ropiens.tistory.com/85
- [https://velog.io/@rockgoat2/Reinforcement-Learning-PPO-알고리즘-리뷰](https://velog.io/@rockgoat2/Reinforcement-Learning-PPO-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EB%A6%AC%EB%B7%B0)

# 1. Introduction

# 2. Background: Policy Optimisation

## 2.1 Policy Gradient Methods

## 2.2 Trust Region Methods

TRPO 논문에서는 특정 constraint 이내에서 objective function(”surrogate” objective)가 최대화 되는 policy로 업데이트를 진행하고, 이를 trust region methods라고 한다. (importance sampling 원리)

$$
\begin{gather}
\max_\theta\hat{\mathbb E}_t\left[ \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)} \hat A_t \right]
\\
\text{subject to }\hat{\mathbb E}_t\left[ \text{KL}[\pi_{\theta_{old}}(\cdot\mid s_t),\pi_\theta(\cdot\mid s_t)] \right]\leq\delta
\end{gather}
$$

그런데 $\theta_{old}$는 policy가 업데이트되기 전의 매개변수이다. 즉, 업데이트 되기 전 매개변수와 크게 차이가 나지 않는 범위 내의 새 매개변수들을 이용해 만든 새 policy인 $\pi_\theta(a_t\mid s_t)$와 $\pi_{\theta_{old}}(a_t\mid s_t)$를 이용해서 importance sampling을 수행한다.

Schulman은 처음 TRPO를 정의할 때 특정 surrogate objective가 policy $\pi$의 성능을 보장하기 위한 lower bound를 형성한다는 개념을 따라 constraint 방법이 아니라 아래와 같이 coefficient $\beta$를 가진 penalty 방법을 제시하였다.

$$
\max_\theta \hat{\mathbb E}_t\left[ \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}\hat A_t - \beta\text{KL}[\pi_{\theta_{old}}(\cdot\mid s_t),\pi_\theta(\cdot\mid s_t)] \right]
$$

그럼에도 constraint 방법을 사용하는 이유는 학습하는 도중에도 특성의 변화가 많아서 고정된 $\beta$를 결정하기 어렵기 때문이다. 

그러므로, TRPO와 같이 성능을 보장할 수 있는 policy optimisation을 first-order algorithm으로 계산하려면 추가적인 개선이 필요하다.

# 3. Clipped Surrogate Objective

Importance sampling에 쓰이는 것처럼, 다음을 정의하자.

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}
$$

여기서, TRPO는 다음의 surrogate function을 최대화한다 (위에 언급됨)

$$
L^{CPI}(\theta)=\hat{\mathbb E}_t\left[ \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}\hat A_t \right]=\hat{\mathbb E}_t\left[ r_t(\theta)\hat A_t \right]
$$

*CPI는 conservative policy iteration의 약자.

만약 여기서 constraint가 따로 존재하지 않는다면 policy update가 큰 스텝으로 진행될 것이고, 그러면 monotonous improvement를 보장할 수 없다. 따라서, policy change에 페널티를 부과한다. ($\pi_{\theta_{old}}$에서 너무 많이 벗어나지 않도록).

이를 위해 clip을 적용한 새로운 surrogate objective는 다음과 같다.

$$
L^{CLIP}(\theta)=\hat {\mathbb E}_t\left[ \min\left(r_t(\theta)\hat A_t, \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t \right)\right]
$$

논문에서는 hyperparameter $\epsilon$을 0.2로 설정했다.

이렇게 하면 $r_t$가 $[1-\epsilon,1+\epsilon]$ 밖으로 벗어날 유인(incentive)이 사라진다. clip된 값과 $r_t(\theta)\hat A_t$를 비교해서 더 작은 값을 취하므로 lower bound를 형성하는 데 의미가 있다.

![스크린샷 2024-07-23 16.02.18.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e32255d7-f1d1-45eb-81e3-06e46c789c7a/6effc2f5-f446-4714-8555-60487b8cb8f7/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-23_16.02.18.png)

이런 식으로 penalty를 부과하는 것. 잘 보면 $L^{CLIP}$이 $L^{CPI}$의 lower bound를 형성하는 것을 알 수 있다.

![스크린샷 2024-07-23 16.05.10.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e32255d7-f1d1-45eb-81e3-06e46c789c7a/203d6346-bb30-4021-a488-9335697c1fb8/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-23_16.05.10.png)

만약 clip을 설정하지 않으면 위 그림의 파란색 선처럼 정책이 너무 과감하게 update될 수 있고, 그러면 importance sampling의 의미가 사라진다.

# 4. Adaptive KL Penalty Coefficient

Clip을 쓰는 대신 다른 방법으로는 KL divergence에 페널티를 부과하는 방법이 있다. 앞에서 고정된 $\beta$는 적절치 않다고 했으니, 여기서는 $\beta$를 adaptive하게 변화시켜주는 것.

$\beta$를 어떻게 정하든 default 값과 그 성능이 크게 차이나지 않는다. 알고리즘이 빨리 adjust하기 때문 

하지만 실험 결과 clipping 방식이 더 뛰어난 것으로 나타남

# 5. Algorithm

policy function과 value function이 parameter를 공유하도록 인공신경망을 설계한다면 policy surrogate와 value function error term을 결합한 loss function을 사용해야 한다. 이 목적 함수는 충분한 탐험을 보장하기 위한 entropy bonus가 추가될 수도 있고, 그 식은 다음과 같다.

$$
L_t^{CLIP+VF+S}(\theta)=\hat{\mathbb E}_t\left[ L_t^{CLIP}(\theta)-c_1L_t^{VF}(\theta)+c_2 S[\pi_\theta](s_t) \right]
$$

여기서 $c_1,c_2$는 계수, $S$는 entropy bonus이고, $L_t^{VF}$는 squared-error loss $(V_\theta(s_t)-V_t^{targ})^2$이다.

에피소드의 길이보다 짧은 $T$개의 timestep만큼 policy를 진행시키고, 여기서 얻은 샘플들로 업데이트를 진행한다. 이 때, Advantage estimator $\hat A$는 timestep $T$ 이후를 관측하지 않는다. 

매 iteration마다, $N$개의 parallel한 actor가 $T$개의 샘플들을 모으고, 이 $NT$개의 샘플에 대해 surrogate loss를 계산, minibatch SGD나 Adam을 이용하여 최적화시킨다.

![스크린샷 2024-07-23 16.26.47.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e32255d7-f1d1-45eb-81e3-06e46c789c7a/af9b1719-1190-4d60-b5c6-b43d847f937f/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-23_16.26.47.png)

# 6. Experiments

## 6.1 Comparison of Surrogate Objectives

## 6.2 Comparison to Other Algorithms in the Continuous Domain

## 6.3 Showcase in the Continuous Domain: Humanoid Running and Steering

## 6.4 Comparison to Other Algorithms on the Atari Domain

# 7. Conclusion

# 8. Acknowledgements
