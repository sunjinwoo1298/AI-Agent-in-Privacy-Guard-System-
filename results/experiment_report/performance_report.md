# PrivMAS Performance & Accuracy Analysis

## Overall Performance Summary
This table shows average performance metrics for each agent count.

|   num_agents |   e2e_ms |   t_inf_ms |   delta_sync_ms |   c_tax_ms |
|-------------:|---------:|-----------:|----------------:|-----------:|
|            1 |     0.52 |       0.06 |            0.00 |       0.46 |
|            2 |     7.06 |       6.31 |            6.36 |       0.76 |
|            3 |     7.61 |       6.85 |            6.91 |       0.76 |
|            4 |    11.82 |      10.32 |           11.20 |       1.49 |
|            5 |    12.42 |      10.85 |           11.77 |       1.58 |

## Overall Accuracy Summary
This table shows average accuracy metrics for each agent count.

|   num_agents |   precision |   recall |   f1_score |
|-------------:|------------:|---------:|-----------:|
|            1 |        0.22 |     0.19 |       0.20 |
|            2 |        0.08 |     0.06 |       0.06 |
|            3 |        0.01 |     0.02 |       0.02 |
|            4 |        0.04 |     0.02 |       0.03 |
|            5 |        0.02 |     0.02 |       0.02 |

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

