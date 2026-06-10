import statistics

import numpy as np
from sqlalchemy import select

from app.cache import CacheClient
from app.database import SessionLocal
from app.models import BenchmarkRun, User
from app.recommender import HybridRecommender


def _top_k_indices(scores: np.ndarray, k: int, exclude_mask: np.ndarray) -> np.ndarray:
    adjusted = scores.copy()
    adjusted[exclude_mask] = -1.0
    return np.argsort(adjusted)[::-1][:k]


def main() -> None:
    cache = CacheClient()
    recommender = HybridRecommender(cache=cache)
    recommender.refresh()

    if recommender.interactions_matrix.size == 0:
        print("No interaction matrix found. Run seed_data first.")
        return

    k = 10
    baseline_relevance: list[float] = []
    hybrid_relevance: list[float] = []
    uncached_latencies: list[float] = []
    cached_latencies: list[float] = []

    with SessionLocal() as db:
        users = db.execute(select(User)).scalars().all()

    for user in users:
        if user.id not in recommender.user_index:
            continue

        user_idx = recommender.user_index[user.id]
        seen = recommender.interactions_matrix[user_idx] > 0

        cf_scores = recommender._normalize(recommender._cf_scores(user_idx))
        content_scores = recommender._normalize(recommender._content_scores(user_idx))

        baseline_idx = _top_k_indices(cf_scores, k, seen)
        baseline_relevance.append(float(np.mean(content_scores[baseline_idx])))

        cache_key = f"recs:{user.external_id}:{k}"
        cache.delete(cache_key)
        hybrid_recs, uncached_latency = recommender.recommend(user.external_id, k)
        _, cached_latency = recommender.recommend(user.external_id, k)

        uncached_latencies.append(uncached_latency)
        cached_latencies.append(cached_latency)

        if hybrid_recs:
            hybrid_indices = np.array(
                [
                    recommender.item_external_ids.index(rec.item_external_id)
                    for rec in hybrid_recs
                    if rec.item_external_id in recommender.item_external_ids
                ]
            )
            if hybrid_indices.size > 0:
                hybrid_relevance.append(float(np.mean(content_scores[hybrid_indices])))

    baseline_rel = float(np.mean(baseline_relevance)) if baseline_relevance else 0.0
    hybrid_rel = float(np.mean(hybrid_relevance)) if hybrid_relevance else 0.0
    rel_gain = ((hybrid_rel - baseline_rel) / baseline_rel * 100.0) if baseline_rel > 0 else 0.0

    baseline_latency = float(statistics.mean(uncached_latencies)) if uncached_latencies else 0.0
    hybrid_latency = float(statistics.mean(cached_latencies)) if cached_latencies else 0.0
    latency_reduction = (
        ((baseline_latency - hybrid_latency) / baseline_latency) * 100.0
        if baseline_latency > 0
        else 0.0
    )

    reached_sub_200ms = hybrid_latency < 200.0

    with SessionLocal() as db:
        run = BenchmarkRun(
            run_name="local_50_users",
            baseline_avg_relevance=baseline_rel,
            hybrid_avg_relevance=hybrid_rel,
            relevance_gain_pct=rel_gain,
            baseline_latency_ms=baseline_latency,
            hybrid_latency_ms=hybrid_latency,
            latency_reduction_pct=latency_reduction,
            reached_sub_200ms=reached_sub_200ms,
        )
        db.add(run)
        db.commit()

    print("Benchmark complete")
    print(f"Baseline relevance: {baseline_rel:.4f}")
    print(f"Hybrid relevance:   {hybrid_rel:.4f}")
    print(f"Relevance gain:     {rel_gain:.2f}%")
    print(f"Uncached latency:   {baseline_latency:.2f} ms")
    print(f"Cached latency:     {hybrid_latency:.2f} ms")
    print(f"Latency reduction:  {latency_reduction:.2f}%")
    print("Sub-200ms target:", "PASS" if reached_sub_200ms else "FAIL")


if __name__ == "__main__":
    main()
