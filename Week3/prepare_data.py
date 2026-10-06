"""Download the official CIFAR archive with validated HTTP byte ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import shutil
import urllib.request
from pathlib import Path

URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
SIZE = 170498071
MD5 = "c58f30108f718f92721af3b95e74349a"


def main():
    directory = Path.cwd() / "data"
    directory.mkdir(exist_ok=True)
    target = directory / "cifar-10-python.tar.gz"
    if target.exists() and target.stat().st_size == SIZE:
        assert hashlib.md5(target.read_bytes()).hexdigest() == MD5
        print("Official CIFAR-10 checksum verified.", flush=True)
        return
    prefix = target.stat().st_size if target.exists() else 0
    assert 0 <= prefix < SIZE
    print(f"Resume official archive at {prefix:,}/{SIZE:,} bytes", flush=True)
    step = (SIZE-prefix+7)//8
    ranges = [(start, min(start+step, SIZE)-1) for start in range(prefix, SIZE, step)]
    def fetch(bounds):
        start, end = bounds
        part = directory / f"cifar.part.{start}.{end}"
        if part.exists() and part.stat().st_size == end-start+1:
            return start, part
        request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=90) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == f"bytes {start}-{end}/{SIZE}"
            with part.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024*1024)
        assert part.stat().st_size == end-start+1
        print(f"Completed range {start:,}-{end:,}", flush=True)
        return start, part
    pieces = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch, bounds) for bounds in ranges]):
            pieces.append(future.result())
    complete = directory / "cifar-10-complete.tmp"
    with complete.open("wb") as out:
        if prefix:
            assert target.stat().st_size == prefix, "Another process is still writing the archive."
            with target.open("rb") as source:
                shutil.copyfileobj(source, out)
        for _, part in sorted(pieces):
            with part.open("rb") as source:
                shutil.copyfileobj(source, out)
    assert complete.stat().st_size == SIZE
    assert hashlib.md5(complete.read_bytes()).hexdigest() == MD5, "Official checksum mismatch"
    complete.replace(target)
    print("Official CIFAR-10 checksum verified.", flush=True)


if __name__ == "__main__":
    main()
