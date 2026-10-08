"""Host-side torch ops per step for one rank, over a range of steps.

    uv run python scripts/host_ops.py <trace_dir> <rank> <first_step> <last_step> [<top>]

Steps are numbered as in scripts/step_profile.py (``_compute_slot_mapping_kernel``
launches on the device). The window is mapped onto the host clock through the
profiler's own connection ids (a host launch and its device task share one), so
no clock alignment is assumed. Reports, for the busiest host thread, inclusive
time per step by op name; nesting makes the totals overlap, so compare ranks
rather than summing rows.
"""

from __future__ import annotations

import glob
import os
import sqlite3
import sys
from collections import defaultdict

from moe_reliability.core.trace_analysis import ascend_rank_outputs


def main(trace_dir: str, rank: int, first: int, last: int, top: int = 25) -> None:
    out = ascend_rank_outputs(trace_dir)[rank]
    db = sqlite3.connect(glob.glob(os.path.join(out, "ascend_pytorch_profiler_*.db"))[0])
    names = dict(db.execute("select id, value from STRING_IDS"))
    step_id = next(i for i, v in names.items() if v.endswith("_compute_slot_mapping_kernel"))
    # Host launches of the step kernel, in order: CANN_API rows whose connection id the step's TASK shares.
    launches = [r[0] for r in db.execute(
        "select a.startNs from TASK t join COMPUTE_TASK_INFO c on c.globalTaskId = t.globalTaskId "
        "join CANN_API a on a.connectionId = t.connectionId where c.name = ? order by t.startNs", (step_id,))]
    t0, t1 = launches[first], launches[last]
    n = last - first
    tid = db.execute("select globalTid from PYTORCH_API where startNs between ? and ? "
                     "group by globalTid order by count(*) desc limit 1", (t0, t1)).fetchone()[0]
    tot, cnt = defaultdict(float), defaultdict(int)
    for name, s, e in db.execute("select name, startNs, endNs from PYTORCH_API where globalTid = ? "
                                 "and startNs >= ? and startNs < ?", (tid, t0, t1)):
        tot[name] += (min(int(e), t1) - int(s)) / 1e6
        cnt[name] += 1
    print(f"rank {rank}, steps {first}-{last}: host step interval {(t1 - t0) / n / 1e6:.1f} ms")
    for name, ms in sorted(tot.items(), key=lambda kv: -kv[1])[:top]:
        print(f"  {ms / n:8.2f} ms/step  {cnt[name] / n:7.1f}/step  {names.get(name, name)[:90]}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), int(a[2]), int(a[3]), *(int(x) for x in a[4:5]))
