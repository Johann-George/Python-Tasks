from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base
from config.db import get_engine

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customer'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone = Column(String(15))
    loyalty_level = Column(String(20), default="Bronze")    

class Order(Base):
    __tablename__ = 'order'
    id = Column(Integer, primary_key=True, autoincrement=True) 
    name = Column(String(50))
    price = Column(Float)
    quantity = Column(Integer)
    cust_id = Column(Integer, ForeignKey('customer.id'))

Base.metadata.create_all(get_engine())