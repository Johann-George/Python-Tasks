from enum import Enum
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Index
from config.db import db, Base

class LoyaltyLevel(Enum):
    BRONZE = "Bronze"
    SILVER = "Silver"
    GOLD = "Gold"

class Customer(Base):
    __tablename__ = 'customer'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone = Column(String(15))
    loyalty_level = Column(String(20), default=LoyaltyLevel.BRONZE)    

class Order(Base):
    __tablename__ = 'order'
    id = Column(Integer, primary_key=True, autoincrement=True) 
    name = Column(String(50))
    price = Column(Float)
    quantity = Column(Integer)
    cust_id = Column(Integer, ForeignKey('customer.id'))

    # Index created for compute_total_spending() func
    __table_args__ = (
        Index('ix_order_cust_price_qty', 'cust_id', 'price', 'quantity'),
    )




Base.metadata.create_all(db.get_engine())