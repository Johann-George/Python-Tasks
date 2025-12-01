from sqlalchemy import Column, Integer, String, Text
from config.db import db, Base

class Flyjac(Base):
    __tablename__ = 'mail_matrix'
    id = Column(Integer, primary_key=True, autoincrement=True)
    location = Column(String, nullable=False)
    users_ai = Column(Text)
    users_si = Column(Text)
    ai_hod = Column(String)
    si_hod = Column(String)
    mail_id_ai_missing = Column(String)
    si = Column(String)


# Base.metadata.create_all(db.get_engine())