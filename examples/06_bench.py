"""6. The full benchmark: every strategy, one workload, honest accounting.

Router tokens are charged to the router. The cascade pays for the cheap attempt
it discards. The baseline is random routing at the SAME strong-model call
fraction -- not always-strong, which any router beats trivially.
"""
from _shared import MODE, client

from dms.bench import run_bench
from dms.report import render
from dms.workload import Workload

report = run_bench(Workload.load(), client(max_spend_usd=None))
print(render(report))
print(f"\n(mode={MODE.value})")
