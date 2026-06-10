from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ab_testing import ABTestingManager
from app.cache import CacheClient
from app.config import settings
from app.database import Base, engine, get_db
from app.models import Interaction, Item, User
from app.recommender import HybridRecommender
from app.schemas import (
    ABAssignmentResponse,
    ABEventIn,
    HealthResponse,
    InteractionCreate,
    ItemCreate,
    RecommendationsResponse,
    RecommendationOut,
    UserCreate,
)


app = FastAPI(title=settings.app_name)
cache = CacheClient()
recommender = HybridRecommender(cache=cache)
ab_manager = ABTestingManager()


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    recommender.refresh()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = "ok" if cache.ping() else "ok-no-redis"
    return HealthResponse(status=status)


@app.post("/users")
def create_user(payload: UserCreate, db: Session = Depends(get_db)) -> dict:
    exists = db.execute(select(User).where(User.external_id == payload.external_id)).scalar_one_or_none()
    if exists:
        return {"message": "already exists", "external_id": payload.external_id}

    user = User(external_id=payload.external_id)
    db.add(user)
    db.commit()
    return {"message": "created", "external_id": payload.external_id}


@app.post("/items")
def create_item(payload: ItemCreate, db: Session = Depends(get_db)) -> dict:
    exists = db.execute(select(Item).where(Item.external_id == payload.external_id)).scalar_one_or_none()
    if exists:
        return {"message": "already exists", "external_id": payload.external_id}

    item = Item(
        external_id=payload.external_id,
        title=payload.title,
        description=payload.description,
    )
    db.add(item)
    db.commit()
    recommender.last_refresh = 0
    return {"message": "created", "external_id": payload.external_id}


@app.post("/interactions")
def add_interaction(payload: InteractionCreate, db: Session = Depends(get_db)) -> dict:
    user = db.execute(select(User).where(User.external_id == payload.user_external_id)).scalar_one_or_none()
    item = db.execute(select(Item).where(Item.external_id == payload.item_external_id)).scalar_one_or_none()
    if user is None or item is None:
        raise HTTPException(status_code=404, detail="user or item not found")

    interaction = Interaction(
        user_id=user.id,
        item_id=item.id,
        event_type=payload.event_type,
        weight=payload.weight,
    )
    db.add(interaction)
    db.commit()
    recommender.last_refresh = 0
    return {"message": "recorded"}


@app.get("/recommendations/{user_external_id}", response_model=RecommendationsResponse)
def get_recommendations(user_external_id: str, k: int = settings.top_k_default) -> RecommendationsResponse:
    recs, latency = recommender.recommend(user_external_id=user_external_id, k=k)
    return RecommendationsResponse(
        user_external_id=user_external_id,
        latency_ms=latency,
        recommendations=[
            RecommendationOut(item_external_id=r.item_external_id, title=r.title, score=r.score)
            for r in recs
        ],
    )


@app.post("/ab/assign/{user_external_id}", response_model=ABAssignmentResponse)
def assign_user_to_variant(user_external_id: str, db: Session = Depends(get_db)) -> ABAssignmentResponse:
    try:
        variant = ab_manager.assign_user(db, user_external_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ABAssignmentResponse(
        user_external_id=user_external_id,
        experiment_name=settings.ab_test_name,
        variant=variant,
    )


@app.post("/ab/event")
def track_ab_event(payload: ABEventIn, db: Session = Depends(get_db)) -> dict:
    try:
        variant = ab_manager.record_event(
            db,
            user_external_id=payload.user_external_id,
            event_type=payload.event_type,
            metadata=payload.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"status": "recorded", "variant": variant}


@app.get("/ab/metrics")
def ab_metrics(db: Session = Depends(get_db)) -> dict:
    return ab_manager.retention_uplift(db)
