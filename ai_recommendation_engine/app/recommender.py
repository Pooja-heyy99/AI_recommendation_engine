import json
import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.cache import CacheClient
from app.config import settings
from app.database import SessionLocal
from app.models import Interaction, Item, User


@dataclass
class RankedItem:
    item_external_id: str
    title: str
    score: float


class HybridRecommender:
    def __init__(self, cache: CacheClient) -> None:
        self.cache = cache
        self.content_weight = settings.content_weight
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.tf_model: tf.keras.Model | None = None
        self.last_refresh = 0.0

        self.user_ids: list[int] = []
        self.item_ids: list[int] = []
        self.user_index: dict[int, int] = {}
        self.item_index: dict[int, int] = {}
        self.item_external_ids: list[str] = []
        self.item_titles: list[str] = []
        self.item_embeddings: np.ndarray = np.zeros((0, 384), dtype=np.float32)
        self.interactions_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def refresh(self) -> None:
        with SessionLocal() as db:
            items = db.execute(select(Item).order_by(Item.id)).scalars().all()
            users = db.execute(select(User).order_by(User.id)).scalars().all()
            interactions = db.execute(select(Interaction)).scalars().all()

            if not items:
                self.last_refresh = time.time()
                return

            descriptions = [it.description for it in items]
            embeds = self.sentence_model.encode(descriptions, normalize_embeddings=True)

            for it, emb in zip(items, embeds, strict=True):
                it.embedding_json = json.dumps(emb.tolist())
            db.commit()

            self.user_ids = [u.id for u in users]
            self.item_ids = [it.id for it in items]
            self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}
            self.item_index = {iid: i for i, iid in enumerate(self.item_ids)}
            self.item_external_ids = [it.external_id for it in items]
            self.item_titles = [it.title for it in items]
            self.item_embeddings = np.array(embeds, dtype=np.float32)
            self.interactions_matrix = np.zeros((len(self.user_ids), len(self.item_ids)), dtype=np.float32)

            for inter in interactions:
                if inter.user_id in self.user_index and inter.item_id in self.item_index:
                    uidx = self.user_index[inter.user_id]
                    iidx = self.item_index[inter.item_id]
                    self.interactions_matrix[uidx, iidx] += float(inter.weight)

            self.tf_model = self._train_tensorflow_reranker()
            self.last_refresh = time.time()

    def _train_tensorflow_reranker(self) -> tf.keras.Model | None:
        if self.interactions_matrix.size == 0 or len(self.user_ids) == 0 or len(self.item_ids) == 0:
            return None

        x_rows: list[list[float]] = []
        y_rows: list[float] = []

        for uidx in range(len(self.user_ids)):
            cf = self._cf_scores(uidx)
            content = self._content_scores(uidx)
            labels = (self.interactions_matrix[uidx] > 0).astype(np.float32)
            for iidx in range(len(self.item_ids)):
                x_rows.append([float(cf[iidx]), float(content[iidx])])
                y_rows.append(float(labels[iidx]))

        x = np.array(x_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.float32)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(2,)),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(x, y, epochs=3, batch_size=32, verbose=0)
        return model

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        minimum = float(np.min(arr))
        maximum = float(np.max(arr))
        span = maximum - minimum
        if span < 1e-9:
            return np.zeros_like(arr)
        return (arr - minimum) / span

    def _cf_scores(self, user_idx: int) -> np.ndarray:
        if self.interactions_matrix.size == 0:
            return np.zeros((len(self.item_ids),), dtype=np.float32)

        user_vector = self.interactions_matrix[user_idx : user_idx + 1]
        sims = cosine_similarity(user_vector, self.interactions_matrix)[0]
        denom = np.sum(np.abs(sims)) + 1e-9
        preds = sims @ self.interactions_matrix / denom
        return preds.astype(np.float32)

    def _content_scores(self, user_idx: int) -> np.ndarray:
        if self.item_embeddings.size == 0:
            return np.zeros((len(self.item_ids),), dtype=np.float32)

        interacted = self.interactions_matrix[user_idx] > 0
        if not np.any(interacted):
            return np.zeros((len(self.item_ids),), dtype=np.float32)

        profile = np.mean(self.item_embeddings[interacted], axis=0, dtype=np.float32)
        profile = profile.reshape(1, -1)
        scores = cosine_similarity(profile, self.item_embeddings)[0]
        return scores.astype(np.float32)

    def _user_id_by_external(self, db: Session, user_external_id: str) -> int | None:
        user = db.execute(select(User).where(User.external_id == user_external_id)).scalar_one_or_none()
        if user is None:
            return None
        return user.id

    def recommend(self, user_external_id: str, k: int) -> tuple[list[RankedItem], float]:
        start = time.perf_counter()
        cache_key = f"recs:{user_external_id}:{k}"
        cached = self.cache.get_json(cache_key)
        if cached:
            latency = (time.perf_counter() - start) * 1000
            return [RankedItem(**it) for it in cached], latency

        if time.time() - self.last_refresh > 20:
            self.refresh()

        with SessionLocal() as db:
            user_id = self._user_id_by_external(db, user_external_id)

        if user_id is None or user_id not in self.user_index or not self.item_ids:
            return [], (time.perf_counter() - start) * 1000

        user_idx = self.user_index[user_id]
        cf = self._normalize(self._cf_scores(user_idx))
        content = self._normalize(self._content_scores(user_idx))

        hybrid = self.content_weight * content + (1.0 - self.content_weight) * cf

        if self.tf_model is not None:
            features = np.stack([cf, content], axis=1)
            reranked = self.tf_model.predict(features, verbose=0).flatten()
            hybrid = 0.5 * hybrid + 0.5 * reranked

        already_seen = self.interactions_matrix[user_idx] > 0
        hybrid[already_seen] = -1.0

        top_idx = np.argsort(hybrid)[::-1][:k]
        recs = [
            RankedItem(
                item_external_id=self.item_external_ids[i],
                title=self.item_titles[i],
                score=float(hybrid[i]),
            )
            for i in top_idx
            if hybrid[i] >= 0
        ]

        serialized = [ri.__dict__ for ri in recs]
        self.cache.set_json(cache_key, serialized)

        latency = (time.perf_counter() - start) * 1000
        return recs, latency
