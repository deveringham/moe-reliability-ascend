## The alarm rule, and localising by accumulation

| group | model | window_requests | consecutive_windows | detection_rate | fires_before_onset | delay_requests_p50 | requests_to_false_alarm | windows | requests | layers_seen | recall | precision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alarm rule | deepseek-v2 | 8 | 1 | 0.5 | 0.5 | 8.0 | 324.0 |  |  |  |  |  |
| alarm rule | deepseek-v2 | 8 | 2 | 0.985 | 0.015 | 16.0 | 1204.0 |  |  |  |  |  |
| alarm rule | deepseek-v2 | 8 | 3 | 1.0 | 0.0 | 24.0 | None |  |  |  |  |  |
| alarm rule | deepseek-v2 | 32 | 1 | 0.66 | 0.34 | 32.0 | 2208.0 |  |  |  |  |  |
| alarm rule | deepseek-v2 | 32 | 2 | 1.0 | 0.0 | 64.0 | 6016.0 |  |  |  |  |  |
| alarm rule | deepseek-v2 | 32 | 3 | 1.0 | 0.0 | 96.0 | None |  |  |  |  |  |
| alarm rule | deepseek-v2 | 128 | 1 | 0.25 | 0.75 | 128.0 | 2944.0 |  |  |  |  |  |
| alarm rule | deepseek-v2 | 128 | 2 | 0.95 | 0.05 | 256.0 | 16896.0 |  |  |  |  |  |
| alarm rule | deepseek-v2 | 128 | 3 | 1.0 | 0.0 | 384.0 | 32512.0 |  |  |  |  |  |
| alarm rule | mixtral | 8 | 1 | 0.76 | 0.24 | 8.0 | 656.0 |  |  |  |  |  |
| alarm rule | mixtral | 8 | 2 | 1.0 | 0.0 | 16.0 | 1752.0 |  |  |  |  |  |
| alarm rule | mixtral | 8 | 3 | 1.0 | 0.0 | 40.0 | None |  |  |  |  |  |
| alarm rule | mixtral | 32 | 1 | 0.38 | 0.62 | 32.0 | 1120.0 |  |  |  |  |  |
| alarm rule | mixtral | 32 | 2 | 0.99 | 0.01 | 64.0 | 8192.0 |  |  |  |  |  |
| alarm rule | mixtral | 32 | 3 | 1.0 | 0.0 | 96.0 | 9632.0 |  |  |  |  |  |
| alarm rule | mixtral | 128 | 1 | 0.31 | 0.69 | 128.0 | 3072.0 |  |  |  |  |  |
| alarm rule | mixtral | 128 | 2 | 0.96 | 0.04 | 256.0 | 31744.0 |  |  |  |  |  |
| alarm rule | mixtral | 128 | 3 | 0.995 | 0.005 | 384.0 | 37696.0 |  |  |  |  |  |
| localisation | deepseek-v2 |  |  |  |  |  |  | 25 | 200 | 26/26 | 1.0 | 0.857 |
| localisation | deepseek-v2 |  |  |  |  |  |  | 50 | 400 | 26/26 | 1.0 | 1.0 |
| localisation | deepseek-v2 |  |  |  |  |  |  | 100 | 800 | 26/26 | 1.0 | 1.0 |
| localisation | deepseek-v2 |  |  |  |  |  |  | 200 | 1600 | 26/26 | 1.0 | 1.0 |
