## What ordinary traffic concentrates to, by corpus (no skew injected)

| model | corpus | requests | pooled | p99_window_1 | p99_window_8 | p99_window_64 | impact_threshold | over_threshold_window_8 |
|---|---|---|---|---|---|---|---|---|
| deepseek-v2 | dolly | 200 | 1.087 | 1.343 | 1.235 | 1.113 | 2.405 | 0.0 |
| deepseek-v2 | mbpp | 200 | 1.146 | 1.253 | 1.188 | 1.155 | 2.405 | 0.0 |
| deepseek-v2 | ultrachat | 200 | 1.071 | 1.288 | 1.169 | 1.099 | 2.405 | 0.0 |
| deepseek-v2 | mmmlu:DE_DE | 200 | 1.137 | 1.228 | 1.168 | 1.146 | 2.405 | 0.0 |
| deepseek-v2 | mixed (6 corpora) | 1200 | 1.079 | 1.323 | 1.151 | 1.104 | 2.405 | 0.0 |
| deepseek-v2 | gsm8k | 200 | 1.124 | 1.219 | 1.149 | 1.131 | 2.405 | 0.0 |
| deepseek-v2 | mmlu | 200 | 1.103 | 1.216 | 1.149 | 1.122 | 2.405 | 0.0 |
| deepseek-v2 | mmmlu:YO_NG | 200 | 1.38 | 1.448 | 1.413 | 1.392 | 2.405 | 0.0 |
| deepseek-v2 | mmmlu:SW_KE | 200 | 1.323 | 1.416 | 1.37 | 1.339 | 2.405 | 0.0 |
| deepseek-v2 | mixed (8 corpora) | 1594 | 1.149 | 1.434 | 1.298 | 1.191 | 2.405 | 0.0 |
| deepseek-v2 | mmmlu:AR_XY | 199 | 1.183 | 1.25 | 1.205 | 1.191 | 2.405 | 0.0 |
| deepseek-v2 | mmmlu:HI_IN | 195 | 1.157 | 1.242 | 1.179 | 1.164 | 2.405 | 0.0 |
| deepseek-v2 | mmmlu:DE_DE | 200 | 1.138 | 1.227 | 1.169 | 1.146 | 2.405 | 0.0 |
| deepseek-v2 | mmmlu:JA_JP | 200 | 1.14 | 1.242 | 1.169 | 1.148 | 2.405 | 0.0 |
| deepseek-v2 | mmmlu:ZH_CN | 200 | 1.12 | 1.239 | 1.163 | 1.132 | 2.405 | 0.0 |
| deepseek-v2 | mmlu | 200 | 1.103 | 1.215 | 1.149 | 1.122 | 2.405 | 0.0 |
| mixtral | dolly | 200 | 1.13 | 1.441 | 1.297 | 1.162 | 1.314 | 0.0033 |
| mixtral | mbpp | 200 | 1.182 | 1.371 | 1.24 | 1.196 | 1.314 | 0.0 |
| mixtral | mmlu | 200 | 1.149 | 1.335 | 1.212 | 1.17 | 1.314 | 0.0 |
| mixtral | gsm8k | 200 | 1.168 | 1.312 | 1.207 | 1.179 | 1.314 | 0.0 |
| mixtral | mmmlu:DE_DE | 200 | 1.137 | 1.329 | 1.2 | 1.159 | 1.314 | 0.0 |
| mixtral | ultrachat | 200 | 1.086 | 1.375 | 1.197 | 1.115 | 1.314 | 0.0 |
| mixtral | mixed (6 corpora) | 1200 | 1.094 | 1.417 | 1.193 | 1.123 | 1.314 | 0.0 |
| mixtral | mmmlu:YO_NG | 200 | 1.469 | 1.548 | 1.505 | 1.482 | 1.314 | 1.0 |
| mixtral | mixed (8 corpora) | 1596 | 1.238 | 1.529 | 1.387 | 1.288 | 1.314 | 0.1 |
| mixtral | mmmlu:SW_KE | 200 | 1.346 | 1.431 | 1.382 | 1.357 | 1.314 | 0.9833 |
| mixtral | mmmlu:HI_IN | 197 | 1.309 | 1.389 | 1.335 | 1.318 | 1.314 | 0.5517 |
| mixtral | mmmlu:AR_XY | 199 | 1.261 | 1.355 | 1.287 | 1.269 | 1.314 | 0.0 |
| mixtral | mmmlu:JA_JP | 200 | 1.225 | 1.356 | 1.264 | 1.235 | 1.314 | 0.0 |
| mixtral | mmmlu:ZH_CN | 200 | 1.21 | 1.362 | 1.249 | 1.219 | 1.314 | 0.0 |
| mixtral | mmlu | 200 | 1.149 | 1.336 | 1.213 | 1.17 | 1.314 | 0.0 |
| mixtral | mmmlu:DE_DE | 200 | 1.136 | 1.324 | 1.201 | 1.159 | 1.314 | 0.0 |
