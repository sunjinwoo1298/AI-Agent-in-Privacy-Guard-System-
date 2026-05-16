# PrivMAS Performance & Accuracy Analysis

## Overall Performance Summary
This table shows average performance metrics for each agent count.

|   num_agents |   e2e_ms |   t_inf_ms |   delta_sync_ms |   c_tax_ms |
|-------------:|---------:|-----------:|----------------:|-----------:|
|            1 |     0.80 |       0.07 |            0.00 |       0.73 |
|            2 |    13.29 |      12.19 |           12.28 |       1.10 |
|            3 |    13.89 |      12.75 |           12.84 |       1.14 |
|            4 |    24.82 |      22.81 |           23.83 |       2.01 |
|            5 |    25.41 |      23.28 |           24.38 |       2.14 |
|            6 |    36.42 |      33.44 |           35.43 |       2.98 |
|            7 |    35.08 |      32.41 |           34.14 |       2.67 |
|            8 |    51.90 |      47.52 |           50.85 |       4.37 |
|            9 |    47.31 |      43.20 |           46.25 |       4.10 |
|           10 |    57.19 |      52.50 |           56.22 |       4.69 |
|           11 |    69.60 |      63.88 |           68.39 |       5.72 |
|           12 |    78.51 |      71.62 |           77.37 |       6.88 |
|           13 |    71.23 |      65.02 |           70.17 |       6.21 |
|           14 |    87.15 |      79.58 |           85.99 |       7.57 |
|           15 |    71.51 |      65.49 |           70.57 |       6.02 |
|           16 |    81.46 |      74.52 |           80.45 |       6.94 |
|           17 |    86.59 |      79.24 |           85.54 |       7.35 |
|           18 |    86.68 |      78.74 |           85.76 |       7.93 |
|           19 |    90.97 |      82.58 |           89.95 |       8.39 |
|           20 |   101.00 |      92.13 |           99.90 |       8.87 |
|           21 |    94.07 |      86.05 |           93.12 |       8.02 |
|           22 |    99.18 |      90.67 |           98.28 |       8.51 |
|           23 |    97.84 |      89.27 |           96.95 |       8.57 |
|           24 |   106.73 |      97.28 |          105.82 |       9.45 |
|           25 |   108.06 |      98.45 |          107.12 |       9.61 |

## Overall Accuracy Summary
This table shows average accuracy metrics for each agent count.

|   num_agents |   precision |   recall |   f1_score |
|-------------:|------------:|---------:|-----------:|
|            1 |        0.26 |     0.29 |       0.28 |
|            2 |        0.14 |     0.11 |       0.12 |
|            3 |        0.15 |     0.15 |       0.15 |
|            4 |        0.14 |     0.17 |       0.15 |
|            5 |        0.16 |     0.17 |       0.17 |
|            6 |        0.13 |     0.14 |       0.14 |
|            7 |        0.14 |     0.17 |       0.16 |
|            8 |        0.10 |     0.16 |       0.13 |
|            9 |        0.10 |     0.16 |       0.12 |
|           10 |        0.07 |     0.11 |       0.09 |
|           11 |        0.08 |     0.12 |       0.09 |
|           12 |        0.05 |     0.10 |       0.07 |
|           13 |        0.07 |     0.11 |       0.09 |
|           14 |        0.06 |     0.12 |       0.08 |
|           15 |        0.10 |     0.18 |       0.13 |
|           16 |        0.07 |     0.14 |       0.10 |
|           17 |        0.08 |     0.14 |       0.10 |
|           18 |        0.07 |     0.14 |       0.10 |
|           19 |        0.10 |     0.19 |       0.13 |
|           20 |        0.06 |     0.11 |       0.08 |
|           21 |        0.10 |     0.15 |       0.12 |
|           22 |        0.07 |     0.11 |       0.09 |
|           23 |        0.12 |     0.15 |       0.14 |
|           24 |        0.07 |     0.10 |       0.08 |
|           25 |        0.11 |     0.13 |       0.12 |

### Key Metrics Explained
- **e2e_ms**: End-to-end latency.
- **t_inf_ms**: Critical Path (slowest agent).
- **delta_sync_ms**: Synchronization Delay.
- **c_tax_ms**: Coordination Tax (`e2e_ms - t_inf_ms`).

## System Scaling
How performance changes as we add agents.

![Performance Scaling](performance_scaling.png)

## Accuracy Scaling
How accuracy changes as we add agents.

![Accuracy Scaling](accuracy_scaling.png)

## Per-Label Accuracy Analysis
Average precision, recall, and F1-score for each PII label, grouped by the number of agents.



## Agent Latency & Strategy Analysis
The plots below show the latency distribution for each strategy and the mix of strategies assigned.

![Latency Distribution](latency_distribution.png)

![Strategy Distribution](strategy_distribution.png)

