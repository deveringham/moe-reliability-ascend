## Where imbalance costs latency (cost is against each run's own balanced arm)

| group | condition | model | npus | batch_achieved | mode | busiest_rank | repeats | tpot_ms | tpot_sd_ms | cost_pct | cost_sd_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| skew strength | one rank hot, every layer | deepseek-v2 | 4 | 512 | eager | 1.142 | 3 | 174.0 | 4.2 | 0.0 | 3.38 |
| skew strength | one rank hot, every layer | deepseek-v2 | 4 | 512 | eager | 1.303 | 3 | 173.9 | 4.4 | -0.03 | 3.48 |
| skew strength | one rank hot, every layer | deepseek-v2 | 4 | 512 | eager | 1.66 | 3 | 171.8 | 1.6 | -1.25 | 2.54 |
| skew strength | one rank hot, every layer | deepseek-v2 | 4 | 512 | eager | 2.345 | 3 | 175.8 | 2.3 | 1.03 | 2.75 |
| skew strength | one rank hot, every layer | deepseek-v2 | 4 | 512 | eager | 3.449 | 3 | 212.6 | 3.7 | 22.22 | 3.61 |
| skew strength | one rank hot, every layer | mixtral | 4 | 512 | eager | 1.178 | 3 | 165.5 | 1.3 | 0.0 | 1.08 |
| skew strength | one rank hot, every layer | mixtral | 4 | 512 | eager | 1.241 | 3 | 167.1 | 1.4 | 0.96 | 1.16 |
| skew strength | one rank hot, every layer | mixtral | 4 | 512 | eager | 1.38 | 3 | 171.0 | 2.0 | 3.32 | 1.43 |
| skew strength | one rank hot, every layer | mixtral | 4 | 512 | eager | 1.761 | 3 | 176.6 | 0.5 | 6.71 | 0.88 |
| skew strength | one rank hot, every layer | mixtral | 4 | 512 | eager | 2.62 | 3 | 193.9 | 1.1 | 17.16 | 1.1 |
| skew strength | one rank hot, every layer | mixtral | 4 | 512 | eager | 3.364 | 3 | 209.8 | 0.8 | 26.77 | 1.08 |
| execution mode | graph mode | deepseek-v2 | 4 | 512 | graph | 1.142 | 3 | 142.3 | 8.0 | 0.0 | 7.92 |
| execution mode | graph mode | deepseek-v2 | 4 | 512 | graph | 1.303 | 3 | 140.5 | 3.8 | -1.26 | 6.13 |
| execution mode | graph mode | deepseek-v2 | 4 | 512 | graph | 1.66 | 3 | 140.1 | 4.2 | -1.58 | 6.25 |
| execution mode | graph mode | deepseek-v2 | 4 | 512 | graph | 2.345 | 3 | 160.3 | 18.7 | 12.61 | 14.56 |
| execution mode | graph mode | deepseek-v2 | 4 | 512 | graph | 3.449 | 3 | 191.2 | 13.1 | 34.36 | 11.86 |
| execution mode | graph mode | mixtral | 4 | 512 | graph | 1.178 | 3 | 146.1 | 10.3 | 0.0 | 9.98 |
| execution mode | graph mode | mixtral | 4 | 512 | graph | 1.241 | 3 | 143.5 | 7.2 | -1.79 | 8.52 |
| execution mode | graph mode | mixtral | 4 | 512 | graph | 1.38 | 3 | 157.4 | 7.4 | 7.68 | 9.14 |
| execution mode | graph mode | mixtral | 4 | 512 | graph | 1.761 | 3 | 152.1 | 10.7 | 4.06 | 10.37 |
| execution mode | graph mode | mixtral | 4 | 512 | graph | 2.62 | 3 | 187.7 | 1.9 | 28.46 | 9.16 |
| execution mode | graph mode | mixtral | 4 | 512 | graph | 3.364 | 3 | 208.9 | 0.4 | 42.96 | 10.09 |
| skew strength | one rank hot, every layer, fine levels | mixtral | 4 | 512 | eager | 1.112 | 3 | 169.2 | 1.4 | 0.0 | 1.19 |
| skew strength | one rank hot, every layer, fine levels | mixtral | 4 | 512 | eager | 1.133 | 3 | 167.4 | 1.5 | -1.09 | 1.23 |
| skew strength | one rank hot, every layer, fine levels | mixtral | 4 | 512 | eager | 1.195 | 3 | 168.8 | 0.6 | -0.25 | 0.9 |
| skew strength | one rank hot, every layer, fine levels | mixtral | 4 | 512 | eager | 1.273 | 3 | 172.5 | 3.8 | 1.95 | 2.4 |
| skew strength | one rank hot, every layer, fine levels | mixtral | 4 | 512 | eager | 1.365 | 3 | 169.5 | 2.2 | 0.15 | 1.55 |
| skew strength | one rank hot, every layer, fine levels | mixtral | 4 | 512 | eager | 1.758 | 3 | 179.1 | 3.0 | 5.83 | 2.0 |
| skew strength | one rank hot, every layer, fine levels | mixtral | 4 | 512 | eager | 2.618 | 3 | 193.7 | 2.3 | 14.45 | 1.64 |
| skew shape | a different rank hot each layer | deepseek-v2 | 4 | 512 | eager | 1.058 | 3 | 165.7 | 12.9 | 0.0 | 10.97 |
| skew shape | a different rank hot each layer | deepseek-v2 | 4 | 512 | eager | 1.73 | 3 | 165.4 | 9.3 | -0.16 | 9.56 |
| skew shape | a different rank hot each layer | deepseek-v2 | 4 | 512 | eager | 2.465 | 3 | 177.3 | 10.4 | 7.0 | 10.39 |
| skew shape | a different rank hot each layer | deepseek-v2 | 4 | 512 | eager | 3.594 | 3 | 174.8 | 5.5 | 5.5 | 8.84 |
| skew shape | a different rank hot each layer | mixtral | 4 | 512 | eager | 1.095 | 3 | 171.7 | 6.8 | 0.0 | 5.57 |
| skew shape | a different rank hot each layer | mixtral | 4 | 512 | eager | 1.348 | 3 | 167.7 | 2.0 | -2.32 | 4.02 |
| skew shape | a different rank hot each layer | mixtral | 4 | 512 | eager | 1.745 | 3 | 176.4 | 3.5 | 2.79 | 4.54 |
| skew shape | a different rank hot each layer | mixtral | 4 | 512 | eager | 2.625 | 3 | 193.2 | 2.4 | 12.53 | 4.64 |
| expert parallelism | 8 ranks, one expert each | mixtral | 8 | 512 | eager | 1.377 | 3 | 158.3 | 9.5 | 0.0 | 8.49 |
| expert parallelism | 8 ranks, one expert each | mixtral | 8 | 512 | eager | 1.526 | 3 | 156.0 | 9.2 | -1.44 | 8.28 |
| expert parallelism | 8 ranks, one expert each | mixtral | 8 | 512 | eager | 2.021 | 3 | 158.3 | 5.9 | 0.04 | 7.08 |
| expert parallelism | 8 ranks, one expert each | mixtral | 8 | 512 | eager | 3.112 | 3 | 167.9 | 5.1 | 6.05 | 7.14 |
| decode batch, strong skew | batch 256 | mixtral | 4 | 256 | eager | 1.178 | 3 | 137.1 | 1.4 | 0.0 | 1.47 |
| decode batch, strong skew | batch 256 | mixtral | 4 | 256 | eager | 2.62 | 3 | 148.2 | 0.6 | 8.08 | 1.21 |
| decode batch, strong skew | batch 512 | mixtral | 4 | 512 | eager | 1.178 | 3 | 174.0 | 0.8 | 0.0 | 0.63 |
| decode batch, strong skew | batch 512 | mixtral | 4 | 512 | eager | 2.62 | 3 | 199.8 | 2.0 | 14.81 | 1.28 |
| decode batch, strong skew | batch 1000 | mixtral | 4 | 1000 | eager | 1.178 | 3 | 262.0 | 20.8 | 0.0 | 11.22 |
| decode batch, strong skew | batch 1000 | mixtral | 4 | 1000 | eager | 2.62 | 3 | 303.0 | 0.5 | 15.64 | 9.18 |
| decode batch, natural strength | batch 64 | mixtral | 4 | 64 | eager | 1.178 | 3 | 112.2 | 0.8 | 0.0 | 1.02 |
| decode batch, natural strength | batch 64 | mixtral | 4 | 64 | eager | 1.241 | 3 | 114.5 | 2.2 | 2.03 | 2.13 |
| decode batch, natural strength | batch 64 | mixtral | 4 | 64 | eager | 1.38 | 3 | 111.0 | 1.0 | -1.14 | 1.17 |
| decode batch, natural strength | batch 128 | mixtral | 4 | 128 | eager | 1.178 | 3 | 121.1 | 3.7 | 0.0 | 4.37 |
| decode batch, natural strength | batch 128 | mixtral | 4 | 128 | eager | 1.241 | 3 | 121.0 | 0.7 | -0.08 | 3.15 |
| decode batch, natural strength | batch 128 | mixtral | 4 | 128 | eager | 1.38 | 3 | 119.2 | 1.2 | -1.62 | 3.19 |
| decode batch, natural strength | batch 256 | mixtral | 4 | 256 | eager | 1.178 | 3 | 134.0 | 1.6 | 0.0 | 1.74 |
| decode batch, natural strength | batch 256 | mixtral | 4 | 256 | eager | 1.241 | 3 | 137.4 | 1.7 | 2.49 | 1.8 |
| decode batch, natural strength | batch 256 | mixtral | 4 | 256 | eager | 1.38 | 3 | 136.8 | 2.1 | 2.09 | 2.0 |
| decode batch, natural strength | batch 512 | mixtral | 4 | 512 | eager | 1.178 | 3 | 177.0 | 3.6 | 0.0 | 2.85 |
| decode batch, natural strength | batch 512 | mixtral | 4 | 512 | eager | 1.241 | 3 | 177.8 | 2.4 | 0.45 | 2.44 |
| decode batch, natural strength | batch 512 | mixtral | 4 | 512 | eager | 1.38 | 3 | 181.5 | 2.8 | 2.51 | 2.61 |
| decode batch, natural strength | batch 1000 | mixtral | 4 | 1000 | eager | 1.178 | 2 | 261.0 | 7.8 | 0.0 | 4.22 |
| decode batch, natural strength | batch 1000 | mixtral | 4 | 1000 | eager | 1.241 | 1 | 262.9 | 0.0 | 0.74 | 3.01 |
| decode batch, natural strength | batch 1000 | mixtral | 4 | 1000 | eager | 1.38 | 1 | 267.6 | 0.0 | 2.55 | 3.06 |
| traffic (no injected skew) | mmlu | mixtral | 4 | 512 | eager | 1.149 | 3 | 159.0 | 1.6 | 0.0 |  |
| traffic (no injected skew) | mmmlu:DE_DE | mixtral | 4 | 512 | eager | 1.136 | 3 | 170.0 | 1.5 | 6.92 |  |
| traffic (no injected skew) | mmmlu:ZH_CN | mixtral | 4 | 512 | eager | 1.21 | 3 | 170.1 | 1.5 | 6.98 |  |
| traffic (no injected skew) | mmmlu:JA_JP | mixtral | 4 | 512 | eager | 1.225 | 3 | 181.0 | 1.1 | 13.84 |  |
| traffic (no injected skew) | mmmlu:AR_XY | mixtral | 4 | 512 | eager | 1.261 | 3 | 202.5 | 2.5 | 27.36 |  |
| traffic (no injected skew) | mmmlu:HI_IN | mixtral | 4 | 512 | eager | 1.309 | 3 | 221.3 | 1.8 | 39.18 |  |
| traffic (no injected skew) | mmmlu:SW_KE | mixtral | 4 | 512 | eager | 1.346 | 3 | 180.3 | 2.7 | 13.4 |  |
| traffic (no injected skew) | mmmlu:YO_NG | mixtral | 4 | 512 | eager | 1.469 | 3 | 210.5 | 0.8 | 32.39 |  |
