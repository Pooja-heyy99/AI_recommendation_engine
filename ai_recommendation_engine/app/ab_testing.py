import hashlib
import json

import boto3
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import ABAssignment, ABEvent, User


class ABTestingManager:
    def __init__(self) -> None:
        self.cloudwatch = None
        if settings.enable_aws_metrics:
            self.cloudwatch = boto3.client("cloudwatch", region_name=settings.aws_region)

    @staticmethod
    def _variant_for_user(user_external_id: str) -> str:
        hashed = hashlib.sha256(user_external_id.encode("utf-8")).hexdigest()
        return "treatment" if int(hashed[:8], 16) % 2 == 1 else "control"

    def assign_user(self, db: Session, user_external_id: str) -> str:
        user = db.execute(select(User).where(User.external_id == user_external_id)).scalar_one_or_none()
        if user is None:
            raise ValueError(f"User not found: {user_external_id}")

        existing = db.execute(
            select(ABAssignment).where(
                ABAssignment.user_id == user.id,
                ABAssignment.experiment_name == settings.ab_test_name,
            )
        ).scalar_one_or_none()
        if existing:
            return existing.variant

        variant = self._variant_for_user(user_external_id)
        row = ABAssignment(
            user_id=user.id,
            experiment_name=settings.ab_test_name,
            variant=variant,
        )
        db.add(row)
        db.commit()
        return variant

    def record_event(self, db: Session, user_external_id: str, event_type: str, metadata: dict) -> str:
        user = db.execute(select(User).where(User.external_id == user_external_id)).scalar_one_or_none()
        if user is None:
            raise ValueError(f"User not found: {user_external_id}")

        variant = self.assign_user(db, user_external_id)
        event = ABEvent(
            user_id=user.id,
            experiment_name=settings.ab_test_name,
            variant=variant,
            event_type=event_type,
            metadata_json=json.dumps(metadata),
        )
        db.add(event)
        db.commit()

        if self.cloudwatch is not None:
            self.cloudwatch.put_metric_data(
                Namespace="AIRecommendationEngine",
                MetricData=[
                    {
                        "MetricName": event_type,
                        "Value": 1.0,
                        "Unit": "Count",
                        "Dimensions": [{"Name": "variant", "Value": variant}],
                    }
                ],
            )

        return variant

    def retention_uplift(self, db: Session) -> dict:
        assigned_rows = db.execute(
            select(ABAssignment.variant, func.count(ABAssignment.id)).where(
                ABAssignment.experiment_name == settings.ab_test_name
            ).group_by(ABAssignment.variant)
        ).all()
        assigned = {variant: count for variant, count in assigned_rows}

        retained_rows = db.execute(
            select(ABEvent.variant, func.count(ABEvent.id)).where(
                ABEvent.experiment_name == settings.ab_test_name,
                ABEvent.event_type == "retained",
            ).group_by(ABEvent.variant)
        ).all()
        retained = {variant: count for variant, count in retained_rows}

        control_rate = (retained.get("control", 0) / max(assigned.get("control", 1), 1))
        treatment_rate = (retained.get("treatment", 0) / max(assigned.get("treatment", 1), 1))

        uplift = 0.0
        if control_rate > 0:
            uplift = ((treatment_rate - control_rate) / control_rate) * 100.0

        return {
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "uplift_pct": uplift,
        }
