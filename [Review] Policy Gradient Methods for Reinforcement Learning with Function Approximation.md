# Policy Gradient Methods for Reinforcement Learning with Function Approximation

- Author: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour
- Journal: NIPS
- Year:  1999
- Link: [[pdf](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)]

> Our main new result is to show that the gradient can be written in a form suitable for estimation from experience aided by an approximate action-value or advantage function. Using this result, we prove for the first time that a version of policy iteration with arbitrary differentiable function approximation is convergent to a locally optimal policy.

## 1. Introduction

In a reinforcement learning setting, we need a function approximator, whether it be neural networks, decision-trees or instance-based methods. So far, it was focused on approximating the value function $Q$. In this framework, the action selection policy is represented implicitly. That is, we select the action with the highest estimated value.

However, the value-function approach comes with several limitations:
1. Value-based methods focus on finding deterministic policies, whereas the optimal policy is often stochastic.
2. An arbitrarily small change in the estimated value of an action can cause it to be, or not be, selected.

Policy gradient, on the other hand, uses a function that **approximates a stochastic policy directly**. For example, suppose a policy is represented by a neural network parameterised by policy parameters $\theta$, whose input is a representation of state $\phi(s)$ and output is the set of probabilities for choosing each action.

Let $\theta$ denote the vector of policy parameters and $rho$ the performance of the corresponding policy (e.g. the average reward per step). In the policy gradient approach, we update the parameters using gradient ascent:

$$
\begin{align}
\Delta \theta\approx \alpha\frac{\partial \rho}{\partial\theta}
\end{align}
$$

where $\alpha$ is a step size. We can see that unlike the value-function approach, small changes in $\theta$ can cause only small changes in the policy.

## 2. Policy Gradient Theorem

We assume that a policy $\pi$ is differentiable with respect to its parameters, i.e. $\frac{\partial\pi(s,a;\theta)}{\partial\theta}$, exists. The expected reward $J=\mathbb E[G_0]$ is defined by:

$$
\begin{align}
J(\theta) = \int_{\mathcal S}\rho^\pi(s)\int_{\mathcal A}\pi_\theta(s,a)r(s,a)~\textrm{d}a\textrm{d}s = \mathbb E_{s\sim \rho^\pi}[r(s,a)].
\end{align}
$$

Then we have

$$
\begin{align}
\nabla_\theta J(\theta)
&= \int_{\mathcal S}\rho^\pi(s)\int_{\mathcal A}\nabla_\theta \pi_\theta(a\mid s)Q^\pi(s,a)\textrm{d}a\textrm{d}s
\\
&= \mathbb E_{s\sim\rho^\pi, a\sim\pi_\theta}[\nabla_\theta\log \pi_\theta(a\mid s)Q^\pi(s,a)]
\end{align}
$$

## 3. Policy Gradient with Approximation

## 4. Application to Deriving Algorithms and Advantages 

## 5. Convergence of Policy Iteration with Function Approximation 
