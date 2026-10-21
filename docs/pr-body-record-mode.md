### What this PR does / why we need it?

Currently with `eplb_config.expert_map_record_path` set EPLB starts one expert-weight transfer, never completes it, and silently stops recording and accurately planning for every subsequent cycle.

The execution sequence:

1. `EplbUpdator.forward_before` generates the D2D task and calls `asyn_expert_weight_transfer` whenever `update_expert_weight_flag()` is true. It does not check the record path, so the loader moves to `TRANSFERRING`.
2. `EplbUpdator.forward_end` skips `update_expert_map_and_weight` when a record path is set.
3. Because `update_expert_map_and_weight` is the only code which waits on P2P requests and returns the loader to `WAITING`, the loader stays in `TRANSFERRING` for the life of the server. The first cycle's `isend`/`irecv` are never waited on, and every later cycle logs `Current D2D weight update is on-going, cannot accept new update task` and returns without accepting the update.

`EplbWorker` is also affected. `do_update` sets `self.old_expert_maps = new_expert_maps` and writes to `shared_dict["expert_maps"]` after every EPLB plan whether or not it was applied, so from the second cycle on it plans against a layout which was never actually executed.

The existing skip in `forward_end` shows that recording was meant not to move weights, so this PR makes that consistent:

- `forward_before` still pops the update info to keep the queue aligned with the cycle but returns before generating or starting the D2D transfer.
- `EplbWorker` takes a `record_only` flag passed through `EplbProcess` from `expert_map_record_path`. When recording it still publishes the plan for export but keeps the installed placement as the basis for the next plan.

The three calls skipped while recording — `set_log2phy_map`, `generate_expert_d2d_transfer_task` and `asyn_expert_weight_transfer` write only to the loader and the loader's `updated_log2phy_map` and queued P2P ops are consumed solely by `update_expert_map_and_weight` which recording mode already skips.

The result is a functioning record mode which records load, plans, and then records the planned expert map without executing any actual expert migration. This is usefule to see what EPLB would do before enabling dynamic migration.

**AI assistance:** this PR was prepared with Claude (Anthropic), which helped partially write the patch code and tests. I have reviewed every changed line.

### Does this PR introduce _any_ user-facing change?

Yes, for recording mode only. Currently EPLB with a record path set silently stops working after the first cycle. After this PR it plans every cycle, records a plan computed from the real layout, and moves no experts. Dynamic EPLB without a record path is unchanged.

### How was this patch tested?

**Unit tests.** `tests/ut/eplb/test_eplb_updator.py::TestEplbUpdatorRecordMode` asserts that `forward_before` starts no transfer while recording and still starts one when not recording. It also tests that and that the update info is consumed from the queue in either case. `tests/ut/eplb/core/test_eplb_worker.py::TestEplbWorkerRecordOnly` asserts that a recording worker plans from the same placement on the second cycle, still publishes the plan for export, and that without `record_only` the plan becomes the new basis as before. All three regression cases fail against unpatched sources. `tests/ut/eplb` is sucessful: 24 passed, against 19 before these patch.

**Reproducing bugs on hardware.** We used DeepSeek-V2-Lite-Chat, 4x Ascend 910B3, policy 2, heat collection interval 50, algorithm execution interval 50. With a record path set, every server logged the "on-going" message from the second cycle on and no weight-update cycle completed in any run. One run made 84 plans and applied none. With the record path unset, the same settings completed 7-12 cycles per point without problem.

**Patched behavior on hardware.** Same model and hardware, 1200 requests at 300 generated tokens, with a router-bias skew giving a 1.72x busiest-rank load so there is something to plan against. Two arms in one run:

| arm | plans | applied cycles | "on-going" seen | busiest rank, by EPLB's own `_compute_imbalance` |
|---|---|---|---|---|
| record path set | 12 | 0 | no | 1.68x -> 1.81x (never falls) |
| no record path | 12 | 12 | no | 1.68x -> 1.02x |

The recording arm plans on every cycle instead of stalling after the first and moves nothing. Its measured imbalance stays at 1.68-1.83x across all twelve plans while the planner keeps proposing the same ~1.3x improvement because every plan is now computed from the placement the server actually implements. The control arm is unaffected and still converges.

---

### Checklist

#### Testing

- [x] **Boundary / edge-case coverage**: the tests cover both settings of the record path and the second planning cycle, which is where the worker's stale basis begins to show.
- [x] **Regression test**: three cases fail without this change.
- [x] **Error / failure path coverage**: the previously broken path — a record path set — is the path under test.
- [x] **Meaningful & reliable tests**: the tests assert what the loader and planner are asked to do, not that a mock was called.
- [ ] **Performance impact**: not applicable. Recording mode moves no weights after this change, and the non-recording path is untouched; no latency claim is made.
