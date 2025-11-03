import os
from sqlmodel import SQLModel, create_engine, Session
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL not found in .env file.")

engine = create_engine(DATABASE_URL, echo=False)

def get_session():
    with Session(engine) as session:
        yield session

def init_db():
    # ✅ Import *all* models here to ensure correct FK registration order
    from backend.models.user import User
    from backend.models.upload import Upload
    from backend.models.prediction import Prediction

    print("🧠 Creating tables (users, uploads, predictions)...")
    SQLModel.metadata.create_all(engine)
    print("✅ Tables created successfully!")
