# -*- coding: utf-8 -*-

import time
import wave

from sev_huey import reg

if __name__ == '__main__':
    file_path = '1-12.wav'
    with wave.open(file_path) as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        assert f.getframerate() == 16000
        data = f.readframes(f.getnframes())
    for i in range(100):
        start = time.time()
        for j in range(4):
            r = reg(data, time.time())
        print(r(blocking=True, timeout=1, backoff=1, max_delay=0.1, revoke_on_timeout=True))
        print(f"Elapsed {time.time() - start:.3}s")
