## What a deployment's own calibration says, and what each alarm margin buys

| model | npus | batch | mode | noise_floor_pct | impact_threshold | benign_ceiling | headroom | margin | alarm | false_alarm_rate | warns_before_cost | catches_once_costly |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v2 | 4 | 512 | eager | 2.75 | 2.434 | 1.142 | 1.292 | None | None | None | None | None |
| mixtral | 4 | 512 | eager | 0.86 | 1.234 | 1.178 | 0.056 | None | None | None | None | None |
| deepseek-v2 | 4 | 512 | eager | 8.95 | None | 1.171 | None | None | None | None | None | None |
| mixtral | 4 | 512 | eager | 4.44 | 1.895 | 1.275 | 0.621 | 0.0 | 1.895 | 0.0 | 0.044 | 1.0 |
| mixtral | 4 | 512 | eager | 4.44 | 1.895 | 1.275 | 0.621 | 0.25 | 1.74 | 0.0 | 0.184 | 1.0 |
| mixtral | 4 | 512 | eager | 4.44 | 1.895 | 1.275 | 0.621 | 0.5 | 1.585 | 0.0 | 0.301 | 1.0 |
| mixtral | 4 | 512 | eager | 4.44 | 1.895 | 1.275 | 0.621 | 0.75 | 1.43 | 0.0 | 0.398 | 1.0 |
| mixtral | 4 | 512 | eager | 4.44 | 1.895 | 1.275 | 0.621 | 1.0 | 1.275 | 0.01 | 0.618 | 1.0 |
| mixtral | 4 | 256 | eager | 1.1 | 1.374 | 1.178 | 0.196 | None | None | None | None | None |
| mixtral | 4 | 512 | eager | 0.51 | 1.227 | 1.178 | 0.049 | None | None | None | None | None |
| mixtral | 4 | 1024 | eager | 9.16 | 2.023 | 1.178 | 0.845 | None | None | None | None | None |
| mixtral | 4 | 512 | eager | 0.95 | 1.42 | 1.278 | 0.142 | 0.0 | 1.42 | 0.0 | 0.066 | 0.998 |
| mixtral | 4 | 512 | eager | 0.95 | 1.42 | 1.278 | 0.142 | 0.25 | 1.385 | 0.0 | 0.098 | 0.999 |
| mixtral | 4 | 512 | eager | 0.95 | 1.42 | 1.278 | 0.142 | 0.5 | 1.349 | 0.0 | 0.162 | 1.0 |
| mixtral | 4 | 512 | eager | 0.95 | 1.42 | 1.278 | 0.142 | 0.75 | 1.314 | 0.0017 | 0.236 | 1.0 |
| mixtral | 4 | 512 | eager | 0.95 | 1.42 | 1.278 | 0.142 | 1.0 | 1.278 | 0.01 | 0.324 | 1.0 |
