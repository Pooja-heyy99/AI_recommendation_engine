import random

from sqlalchemy import select

from app.ab_testing import ABTestingManager
from app.database import SessionLocal
from app.models import User


def main() -> None:
    manager = ABTestingManager()

    with SessionLocal() as db:
        users = db.execute(select(User)).scalars().all()
        if not users:
            print("No users found. Run seed_data first.")
            return

        for user in users:
            manager.assign_user(db, user.external_id)

            variant = manager.assign_user(db, user.external_id)
            retained_probability = 0.30 if variant == "control" else 0.366
            if random.random() < retained_probability:
                manager.record_event(db, user.external_id, "retained", {"window_days": 7})

        metrics = manager.retention_uplift(db)

    print("A/B Test Metrics")
    print(f"Control retention rate:   {metrics['control_rate']:.4f}")
    print(f"Treatment retention rate: {metrics['treatment_rate']:.4f}")
    print(f"Uplift:                   {metrics['uplift_pct']:.2f}%")


if __name__ == "__main__":
    main()
