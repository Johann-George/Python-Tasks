import asyncpg
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import Settings

Base = declarative_base()


async def test_connection():
    try:
        set_instance = Settings()
        db_url = set_instance.database_url()
        conn = await asyncpg.connect(db_url)
        print(f"Successfully connected to the database")
        await conn.close()
    except Exception as e:
        print("DB connection failed:", e)


class Database:

    def __init__(self):
        self.settings = Settings()
        self._engine = None
        self._SessionLocal = None

    def get_engine(self):
        if self._engine is None:
            print("Loaded settings:", self.settings.model_dump())
            self._engine = create_async_engine(self.settings.database_url())
        return self._engine

    def get_session(self):
        if self._SessionLocal is None:
            engine = self.get_engine()
            self._SessionLocal = async_sessionmaker(
                bind=engine, expire_on_commit=False
            )
        return self._SessionLocal()


db = Database()