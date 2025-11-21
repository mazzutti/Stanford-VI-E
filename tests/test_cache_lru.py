# mypy: ignore-errors
import threading
import time

from src.utils.lru import LRUCache


def test_lru_basic_eviction():
    c = LRUCache[int](maxsize=2)
    c.set("a", 1)
    c.set("b", 2)
    assert c.keys() == ["a", "b"]
    # access a -> order becomes b,a
    assert c.get("a") == 1
    assert c.keys() == ["b", "a"]
    # insert c -> evict oldest (b)
    c.set("c", 3)
    assert c.keys() == ["a", "c"]


def test_lru_concurrent_smoke():
    c = LRUCache[int](maxsize=100)

    def writer(start):
        for i in range(start, start + 100):
            c.set(f"k{i}", i)

    def reader():
        for _ in range(100):
            # random reads
            for k in list(c.keys())[:10]:
                _ = c.get(k)
            time.sleep(0.001)

    threads = []
    for s in range(5):
        t = threading.Thread(target=writer, args=(s * 100,))
        threads.append(t)
        t.start()

    r = threading.Thread(target=reader)
    r.start()

    for t in threads:
        t.join()
    r.join()

    # basic sanity: no exceptions and keys present
    assert len(c.keys()) <= 100
