from typing import Any

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    external_id: str = Field(min_length=1, max_length=64)


class ItemCreate(BaseModel):
    external_id: str = Field(min_length=1, max_length=64)
    title: str
    description: str


class InteractionCreate(BaseModel):
    user_external_id: str
    item_external_id: str
    event_type: str = "view"
    weight: float = 1.0


class RecommendationOut(BaseModel):
    item_external_id: str
    title: str
    score: float


class RecommendationsResponse(BaseModel):
    user_external_id: str
    latency_ms: float
    recommendations: list[RecommendationOut]


class ABAssignmentResponse(BaseModel):
    user_external_id: str
    experiment_name: str
    variant: str


class ABEventIn(BaseModel):
    user_external_id: str
    event_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
