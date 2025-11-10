from model.models import Order 
from config.db import db

class OrderService:

    @staticmethod
    def create_order(name: str, price: float, quantity: int, cust_id: int):
        """
            Saves the orders information to the order table

            Arguments:
                name: The name of the customer who performed the order
                price: The price of the item ordered
                quantity: The number of that item ordered
                cust_id: The id of the customer
            
            Returns:
                The order object of the updated order
            
            Raises:
                Exception: Any unexpected errors
        """
        try:
            with db.get_session() as session:
                new_order = Order(name=name, price=price, quantity=quantity, cust_id=cust_id)
                session.add(new_order)
                session.commit()
                session.refresh(new_order)
                return new_order
        except Exception as e:
            print("An unexpected error occured:", e)

        
    @staticmethod
    def update_order(order_id: int, name: str = None, price: float = None, quantity: int = None):
        """
            Updates the orders information to the order table

            Arguments:
                name: The name of the customer who performed the order
                price: The price of the item ordered
                quantity: The number of that item ordered
            
            Returns:
                The order object of the new order
            
            Raises:
                Exception: Any unexpected errors
        """
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
        

    @staticmethod
    def get_order_by_id(order_id: int):
        """
            Gets the order info based on the order ID

            Arguments:
                order_id: The ID of the order
            
            Returns:
                The query object of the order based on the corresponding order_id
            
            Raises:
                Exception: Any unexpected errors
        """
        try:
            with db.get_session() as session:
                return session.query(Order).filter(Order.id == order_id).first()
        except Exception as e:
            print("An unexpected error occured:", e)


