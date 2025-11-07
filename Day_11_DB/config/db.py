from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config.settings import Settings

class Database:
    """
    Centralized database manager that handles engine creation, session management,
    and connection testing. Avoids use of global variables.
    """

    def __init__(self):
        self.settings = Settings()
        self._engine = None
        self._SessionLocal = None

    def get_engine(self):
        """Create (or reuse) the SQLAlchemy engine."""
        if self._engine is None:
            print("Loaded settings:", self.settings.model_dump())  # Debug line
            self._engine = create_engine(self.settings.database_url())
        return self._engine

    def get_session(self):
        """Create a new session from sessionmaker."""
        if self._SessionLocal is None:
            engine = self.get_engine()
            self._SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind = engine
            )
        return self._SessionLocal()

    def test_connection(self):
        """Check if the database connection is successful."""
        engine = self.get_engine()
        try:
            with engine.connect() as connection:
                print("Successfully connected to the database")
        except Exception as e:
            print("DB connection failed:", e)

db = Database()