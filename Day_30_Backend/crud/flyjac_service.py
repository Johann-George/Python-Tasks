from config.db import db
from model.models import Flyjac

class FlyjacService:

    @staticmethod
    async def get_info(id: int = None):
        try:
            async with db.get_session() as session:
                if id:
                    return await session.query(Flyjac).filter(Flyjac.id == id).first()
                return await session.query(Flyjac).all()
        except Exception as e:
            print("An unexpected error occurred get_info():", e)