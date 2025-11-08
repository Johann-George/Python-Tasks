from model.models import Customer, Order
from config.db import db
from sqlalchemy import func

class Customer:

    def create_customer(name: str, email: str, phone: str = None):
        try:
            with db.get_session() as session:
                new_customer = Customer(name=name, email=email, phone=phone)
                session.add(new_customer)
                session.commit()
                session.refresh(new_customer)
                return new_customer
        except Exception as e:
            print("An unexpected error occured:",e)


    def get_customer_details(customer_id: int=None):
        try:
            with db.get_session() as session:
                if customer_id:
                    return session.query(Customer).filter(Customer.id == customer_id).first()
                return session.query(Customer).all()
        except Exception as e:
            print("An unexpected error occured:",e)


    # def get_all_customers():
    #     with get_session() as session:
    #         return session.query(Customer).all()
        
    def update_customer(customer_id: int, name: str = None, email: str = None, phone: str = None, loyalty_level: str = None):
        try:
            with db.get_session() as session:
                customer = session.query(Customer).filter(Customer.id == customer_id).first()
                if not customer:
                    return None
                if name:
                    customer.name = name
                if email:
                    customer.email = email
                if phone:
                    customer.phone = phone
                if loyalty_level:
                    customer.loyalty_level = loyalty_level
                session.commit()
                session.refresh(customer)
                return customer
        except Exception as e:
            print("An unexpected error occured:",e)

    def delete_customer(customer_id: int):
        try:
            with db.get_session() as session:
                customer = session.query(Customer).filter(Customer.id == customer_id).first()
                if not customer:
                    return False
                session.delete(customer)
                session.commit()
                return True
        except Exception as e:
            print("An unexpected error occured:",e)


    def fetch_all_customers_with_total_spending():
        try:
            with db.get_session() as session:
                results = ( 
                    session.query(
                        Customer.id.label("Customer ID"),
                        Customer.name.label("Customer Name"),
                        Customer.email.label("Email"),
                        Customer.phone.label("Phone"),
                        func.coalesce(func.sum(Order.price * Order.quantity), 0).label("Total Spending")
                    )
                    .outerjoin(Order, Customer.id == Order.cust_id)
                    .group_by(Customer.id, Customer.name, Customer.email, Customer.phone)
                    .order_by(func.sum(Order.price * Order.quantity).desc())
                    .all()
                )

                return results
        except Exception as e:
            print("An unexpected error occured:",e)


    def compute_total_spending():
        try:
            with db.get_session() as session:
                spending_data = (
                    session.query(
                        Customer.id,
                        func.coalesce(func.sum(Order.price * Order.quantity), 0).label("total_spending")
                    )
                    .outerjoin(Order, Customer.id == Order.cust_id)
                    .group_by(Customer.id)
                    .all()
                )
                return spending_data
        except Exception as e:
            print("An unexpected error occured:",e)
