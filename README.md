# Learning Trading Strategies

## Introduction
This repository has examples of optimal design of trading strategies using (offline) machine learning and reinforcement-learning techniques.

Investment universe consists of $N$ assets (stocks, bonds, cryptos, currencies, futures, etc). $P_{i,t}$ is the execution price of asset $i$ at time $t$. This is typically assumed to be the closing price for the ease of analysis. Let $K_t$ be the dollar value of the portfolio at time $t$ and $w_{i,t}$ be the fraction of wealth invested in asset $i$ at time $t$ (also known as allocation or weight). A trading strategy is essentially a specification of $w_{i,t}$. We compute portfolio value or strategy net asset value (NAV) at any time as:

$$ K_t = K_{t-1} (1 + \sum_{i=1}^N (w_{i,t-1} r_{i,t} - \vert w_{i,t}-w_{i,t-1} \vert c_i))$$

where $r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1$ is one period return and $c_i$ is the transaction cost for trading asset $i$. Typically we set $K_0=1$.

Equivalently, one period return of this strategy is

$$ r_{p,t} = \sum_{i=1}^N (w_{i,t-1} r_{i,t} - \vert w_{i,t}-w_{i,t-1} \vert c_i) $$

When short-selling is not allowed, we impose constraints on weights at each time $t$: $w_{i,t}\geq 0$ and $\sum_i w_{i,t} = 1$.

The key idea is to parametrize $w_t$ with some differentiable function i.e. $w_t = f(x_t; \theta)$, where $x_t$ is some "feature" vector (e.g. past returns of assets) and $\theta$ are parameters of the function (e.g. weights of a MLP). We then compute strategy returns and, using these returns, a suitable performance measure.

## Total Return Objective
Let's take the total return over a given horizon as performance measure. Let the investment horizon be $T$ periods.

$$ \mathbb{J}(\theta) = E \left( \sum_{t=1}^T r_{p,t} \right) = E \left( \sum_{t=1}^T \sum_{i=1}^N (f_i(x_{t-1};\theta) r_{i,t} - \vert f_i(x_t;\theta)-f_i(x_{t-1};\theta) \vert c_i) \right) $$

where $f_i$ is the $i^{th}$ component of $f$. In SGD framework, we can replace the expectation with sample average and in RL (Robins-Monro) setting, with a single path.

All we need to maximize $\mathbb{J}(\theta)$ is gradient of $\mathbb{J}(\theta)$: $\theta \leftarrow \theta + \eta \nabla_\theta\mathbb{J}$.


## Mean Variance Objective
Total return objective, although useful in RL setting due to its additive nature, ignores risk of the trading strategy. Mean-Variance utility, a popular performance measure in finance, balances risk and return. Here, variance of returns is the measure of risk. The objective function can be written as

$$ \mathbb{J}(\theta) = \text{E}(r_{p,t}) - \frac{\gamma}{2} \text{Var}(r_{p,t}) $$

where $\gamma$ is the risk aversion parameter. Note when $\gamma=0$ (risk-neutral) this measure corresponds to total return. Mean and variance are replaced with their sample equivalents computed over a given investment horizon (i.e. a path)

MVO is not additive in the sense of $G_t = G_{t-1} + r_t$. However, we can approximate it. Let $A_t$ and $B_t$ be mean and second moment of strategy returns at time $t$. We use an exponential averaging scheme for updating:

$$A_t = \delta R_t + (1-\delta) A_{t-1}$$
$$B_t = \delta R_t^2 + (1-\delta) B_{t-1}$$

where $A_0 = 0, B_0=0$. Then,

$$\mathbb{J}_t = A_t - \frac{\gamma}{2} (B_t-A_t^2) $$

It can be shown that

$$\mathbb{J}_t \approx \mathbb{J}_{t-1} + \Delta A_t - \frac{\gamma}{2} (\Delta B_t - 2A_t\Delta A_t)$$

where, $\Delta A_t = R_t - A_{t-1}$ and $\Delta B_t = R_t^2 - B_{t-1}$.

This enables us to run standard RL algorithms with $\Delta A_t - \frac{\gamma}{2} (\Delta B_t - 2A_t\Delta A_t)$ as reward for action taken at time $t$.
