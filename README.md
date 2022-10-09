# ML/RL Applications in Finance

## Introduction
This repository has examples of optimal design of trading strategies using (offline) machine learning and reinforcement-learning techniques.

Suppose our investment universe consists of $N$ assets (stocks, bonds, cryptos, currencies, futures, etc). $P_{i,t}$ is the execution price of asset $i$ at time $t$. This is typically assumed to be the closing price for the ease of analysis. However, VWAP or volume weighted average price is more representative of execution price in practice. Let $K_t$ be the dollar value of the portfolio at time $t$ and $w_{i,t}$ is the fraction of wealth invested in asset $i$ at time $t$ (typically known as allocation or weight). Then, a trading strategy is defined as follows:

$$ K_t = K_{t-1} (1 + \sum_{i=1}^N (w_{i,t-1} r_{i,t} - \vert w_{i,t}-w_{i,t-1} \vert c_i)$$

where $r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1$ is one period return and $c_i$ is the transaction cost for trading asset $i$. Typically we set $K_0=1$.
