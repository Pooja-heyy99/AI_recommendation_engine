import random

from sqlalchemy import select

from app.database import Base, SessionLocal, engine
from app.models import Interaction, Item, User


def seed_users(db, count: int = 50) -> None:
    for i in range(count):
        external_id = f"user_{i+1:03d}"
        existing = db.execute(select(User).where(User.external_id == external_id)).scalar_one_or_none()
        if existing is None:
            db.add(User(external_id=external_id))


def seed_items(db, count: int = 200) -> None:
    categories = ["ml", "cloud", "data", "devops", "nlp", "backend"]
    for i in range(count):
        external_id = f"item_{i+1:04d}"
        existing = db.execute(select(Item).where(Item.external_id == external_id)).scalar_one_or_none()
        if existing is not None:
            continue

        category = random.choice(categories)
        title = f"{category.upper()} Recommendation Asset {i+1}"
        description = (
            f"A {category} focused content item covering practical techniques, "
            f"performance optimization, and production architecture patterns."
        )
        db.add(Item(external_id=external_id, title=title, description=description))


def seed_interactions(db, per_user: int = 20) -> None:
    users = db.execute(select(User)).scalars().all()
    items = db.execute(select(Item)).scalars().all()
    if not users or not items:
        return

    for user in users:
        sampled = random.sample(items, min(per_user, len(items)))
        for item in sampled:
            weight = random.choice([1.0, 1.5, 2.0, 3.0])
            event_type = random.choice(["view", "click", "like"])
            db.add(
                Interaction(
                    user_id=user.id,
                    item_id=item.id,
                    event_type=event_type,
                    weight=weight,
                )
            )


def main() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_users(db, count=50)
        seed_items(db, count=200)
        db.commit()
        seed_interactions(db, per_user=20)
        db.commit()

    print("Seed complete: 50 users, 200 items, and interaction history created.")


if __name__ == "__main__":
    main()
