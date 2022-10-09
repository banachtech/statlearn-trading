# ML/RL Applications in Finance

## Introduction
This repository has examples of optimal design of trading strategies using (offline) machine learning and reinforcement-learning techniques.

Suppose our investment universe consists of $N$ assets (stocks, bonds, cryptos, currencies, futures, etc). $P_{i,t}$ is the execution price of asset $i$ at time $t$. This is typically assumed to be the closing price for the ease of analysis. However, VWAP or volume weighted average price is more representative of execution price in practice. Let $K_t$ be the dollar value of the portfolio at time $t$ and $w_{i,t}$ is the fraction of wealth invested in asset $i$ at time $t$ (typically known as allocation or weight). Then, a trading strategy is defined as follows:

$$ K_t = K_{t-1} (1 + \sum_{i=1}^N (w_{i,t-1} r_{i,t} - \vert w_{i,t}-w_{i,t-1} \vert c_i))$$

where $r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1$ is one period return and $c_i$ is the transaction cost for trading asset $i$. Typically we set $K_0=1$.

Equivalently, one period return of this strategy can be stated as

$$ r_{p,t} = \sum_{i=1}^N (w_{i,t-1} r_{i,t} - \vert w_{i,t}-w_{i,t-1} \vert c_i) $$

When short-selling is not allowed, we impose constraints on weights at each time $t$: $w_{i,t}\geq 0$ and $\sum_i w_{i,t} = 1$.

## Optimization
The key idea is to parametrize $w_t$ with some function i.e. $w_t = f(x_t; \theta)$, where $x_t$ is some "feature" vector (e.g. past returns of assets) and $\theta$ are parameters of the function (e.g. weights of a MLP).

Let's assume the simple measure that is amenable to RL application (i.e. additive): total return over a given horizon. Let the investment horizon be $T$ periods. Then we would like to maximize

$$ \mathbb{J}(\theta) = E \left( \sum_{t=1}^T r_{p,t} \right) = E \left( \sum_{t=1}^T \sum_{i=1}^N (f_i(x_{t-1};\theta) r_{i,t} - \vert f_i(x_t;\theta)-f_i(x_{t-1};\theta) \vert c_i) \right) $$

where the summation is over $T$-length return paths and $f_i$ is the $i^{th}$ component of $f$. In SGD framework, we can replace the expectation with sample average and in RL (Robins-Monro) setting, with a single path.

All we need is gradient of $\mathbb{J}(\theta)$ to apply gradient descent:

$$\theta \Leftarrow \theta + \eta \nabla_\theta\mathbb{J}$$
