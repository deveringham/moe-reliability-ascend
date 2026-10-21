## The alarm rule, and localising by accumulation

| group | model | window_requests | consecutive_windows | detection_rate | fires_before_onset | delay_requests_p50 | requests_to_false_alarm | windows | requests | layers_seen | recall | precision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alarm rule | deepseek-v2 | 8 | 1 | 0.88 | 0.12 | 8.0 | 1012.0 |  |  |  |  |  |
| alarm rule | deepseek-v2 | 8 | 2 | 1.0 | 0.0 | 16.0 | 3120.0 |  |  |  |  |  |
| alarm rule | deepseek-v2 | 8 | 3 | 1.0 | 0.0 | 24.0 | None |  |  |  |  |  |
| alarm rule | deepseek-v2 | 32 | 1 | 0.825 | 0.175 | 32.0 | 4352.0 |  |  |  |  |  |
| alarm rule | deepseek-v2 | 32 | 2 | 0.995 | 0.005 | 64.0 | 9936.0 |  |  |  |  |  |
| alarm rule | deepseek-v2 | 32 | 3 | 1.0 | 0.0 | 96.0 | None |  |  |  |  |  |
| alarm rule | deepseek-v2 | 128 | 1 | 0.565 | 0.435 | 128.0 | 7424.0 |  |  |  |  |  |
| alarm rule | deepseek-v2 | 128 | 2 | 0.985 | 0.015 | 256.0 | 34112.0 |  |  |  |  |  |
| alarm rule | deepseek-v2 | 128 | 3 | 1.0 | 0.0 | 384.0 | None |  |  |  |  |  |
| alarm rule | mixtral | 8 | 1 | 0.63 | 0.37 | 8.0 | 616.0 |  |  |  |  |  |
| alarm rule | mixtral | 8 | 2 | 0.99 | 0.01 | 24.0 | 1704.0 |  |  |  |  |  |
| alarm rule | mixtral | 8 | 3 | 1.0 | 0.0 | 40.0 | None |  |  |  |  |  |
| alarm rule | mixtral | 32 | 1 | 0.395 | 0.605 | 32.0 | 1056.0 |  |  |  |  |  |
| alarm rule | mixtral | 32 | 2 | 0.99 | 0.01 | 64.0 | 6528.0 |  |  |  |  |  |
| alarm rule | mixtral | 32 | 3 | 1.0 | 0.0 | 96.0 | 6304.0 |  |  |  |  |  |
| alarm rule | mixtral | 128 | 1 | 0.335 | 0.665 | 128.0 | 2880.0 |  |  |  |  |  |
| alarm rule | mixtral | 128 | 2 | 0.985 | 0.015 | 256.0 | 23616.0 |  |  |  |  |  |
| alarm rule | mixtral | 128 | 3 | 1.0 | 0.0 | 384.0 | 19200.0 |  |  |  |  |  |
| localisation | deepseek-v2 |  |  |  |  |  |  | 25 | 200 | 26/26 | 1.0 | 0.857 |
| localisation | deepseek-v2 |  |  |  |  |  |  | 50 | 400 | 26/26 | 1.0 | 1.0 |
| localisation | deepseek-v2 |  |  |  |  |  |  | 100 | 800 | 26/26 | 1.0 | 1.0 |
| localisation | deepseek-v2 |  |  |  |  |  |  | 200 | 1600 | 26/26 | 1.0 | 1.0 |
