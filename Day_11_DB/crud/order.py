from model.models import Order 
from config.db import db

class Order:

    def create_order(name: str, price: float, quantity: int, cust_id: int):
        try:
            with db.get_session() as session:
                new_order = Order(name=name, price=price, quantity=quantity, cust_id=cust_id)
                session.add(new_order)
                session.commit()
                session.refresh(new_order)
                return new_order
        except Exception as e:
            print("An unexpected error occured:", e)

        
    def update_order(order_id: int, name: str = None, price: float = None, quantity: int = None):
        try:
            with db.get_session() as session:
                order = session.query(Order).filter(Order.id == order_id).first()
                if not order:
                    return None
                if name:
                    order.name = name
                if price:
                    order.price = price 
                if quantity:
                    order.quantity = quantity
                session.commit()
                session.refresh(order)
                return order
        except Exception as e:
            print("An unexpected error occured:", e)
        

    def get_order_by_id(order_id: int):
        try:
            with db.get_session() as session:
                return session.query(Order).filter(Order.id == order_id).first()
        except Exception as e:
            print("An unexpected error occured:", e)


