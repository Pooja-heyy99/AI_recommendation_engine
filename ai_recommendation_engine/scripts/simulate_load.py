import asyncio
import random
import statistics
import time

import httpx


BASE_URL = "http://localhost:8000"
CONCURRENT_USERS = 50
REQUESTS_PER_USER = 10


async def worker(client: httpx.AsyncClient, user_external_id: str, latencies: list[float]) -> None:
    for _ in range(REQUESTS_PER_USER):
        t0 = time.perf_counter()
        response = await client.get(f"{BASE_URL}/recommendations/{user_external_id}?k=10")
        response.raise_for_status()
        local_latency_ms = (time.perf_counter() - t0) * 1000
        api_latency_ms = response.json().get("latency_ms", local_latency_ms)
        latencies.append(api_latency_ms)
        await asyncio.sleep(random.uniform(0.01, 0.05))


async def main() -> None:
    latencies: list[float] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            worker(client, f"user_{idx+1:03d}", latencies)
            for idx in range(CONCURRENT_USERS)
        ]
        await asyncio.gather(*tasks)

    p50 = statistics.quantiles(latencies, n=100)[49]
    p95 = statistics.quantiles(latencies, n=100)[94]
    avg = sum(latencies) / max(len(latencies), 1)

    print(f"Total requests: {len(latencies)}")
    print(f"Average latency: {avg:.2f} ms")
    print(f"P50 latency: {p50:.2f} ms")
    print(f"P95 latency: {p95:.2f} ms")
    print("Sub-200ms target:", "PASS" if p95 < 200 else "FAIL")


if __name__ == "__main__":
    asyncio.run(main())
