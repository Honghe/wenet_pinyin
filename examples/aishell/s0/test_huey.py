# -*- coding: utf-8 -*-

import time

from sev_huey import reg

for i in range(100):
    start = time.time()
    for j in range(4):
        r = reg('', time.time())
    print(r(blocking=True, timeout=1, backoff=1, max_delay=0.1, revoke_on_timeout=True))
    print(f"elapse {time.time() - start:.3}s")
