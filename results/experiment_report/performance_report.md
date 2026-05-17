# PrivMAS Performance & Accuracy Analysis

## Overall Performance Summary
This table shows average performance metrics for each agent count.

|   num_agents |   e2e_ms |   t_inf_ms |   delta_sync_ms |   c_tax_ms |
|-------------:|---------:|-----------:|----------------:|-----------:|
|            1 |     0.44 |       0.04 |            0.00 |       0.39 |
|            2 |     7.71 |       6.72 |            6.79 |       0.99 |
|            3 |     7.82 |       6.87 |            6.93 |       0.96 |
|            4 |    13.59 |      12.17 |           12.89 |       1.43 |
|            5 |    13.93 |      12.53 |           13.19 |       1.40 |

## Overall Accuracy Summary
This table shows average accuracy metrics for each agent count.

|   num_agents |   precision |   recall |   f1_score |
|-------------:|------------:|---------:|-----------:|
|            1 |        0.26 |     0.29 |       0.28 |
|            2 |        0.14 |     0.11 |       0.13 |
|            3 |        0.15 |     0.15 |       0.15 |
|            4 |        0.13 |     0.17 |       0.15 |
|            5 |        0.18 |     0.17 |       0.18 |

## Leakage Summary
Leakage is reported separately for strict and overlap matching.

|   num_agents |   strict_leakage_rate |   overlap_leakage_rate |   leakage_rate |
|-------------:|----------------------:|-----------------------:|---------------:|
|            1 |                  0.65 |                   0.76 |           0.76 |
|            2 |                  0.93 |                   0.92 |           0.92 |
|            3 |                  0.91 |                   0.90 |           0.90 |
|            4 |                  0.92 |                   0.89 |           0.89 |
|            5 |                  0.97 |                   0.89 |           0.89 |

### Key Metrics Explained
- **e2e_ms**: End-to-end latency.
- **t_inf_ms**: Critical Path (slowest agent).
- **delta_sync_ms**: Synchronization Delay.
- **c_tax_ms**: Coordination Tax (`e2e_ms - t_inf_ms`).

- **strict_leakage_rate**: Fraction of ground-truth PII missed under strict matching.
- **overlap_leakage_rate**: Fraction of ground-truth PII missed under overlap matching.

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

