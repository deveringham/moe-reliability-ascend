## What a deployment's own calibration says, and what each alarm margin buys

| model | npus | batch | mode | noise_floor_pct | impact_threshold | benign_ceiling | headroom | margin | alarm | false_alarm_rate | warns_before_cost | catches_once_costly |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v2 | 4 | 512 | eager | 2.75 | 2.434 | 1.142 | 1.292 | None | None | None | None | None |
| mixtral | 4 | 512 | eager | 0.86 | 1.234 | 1.178 | 0.056 | None | None | None | None | None |
| mixtral | 4 | 256 | eager | 1.1 | 1.374 | 1.178 | 0.196 | None | None | None | None | None |
| mixtral | 4 | 512 | eager | 0.51 | 1.227 | 1.178 | 0.049 | None | None | None | None | None |
| mixtral | 4 | 1024 | eager | 9.16 | 2.023 | 1.178 | 0.845 | None | None | None | None | None |
