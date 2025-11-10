from model.models import Customer, Order
from config.db import db
from sqlalchemy import func

class CustomerService:

    @staticmethod
    def create_customer(name: str, email: str, phone: str = None):
        """
            Saves the customer information to the customer table

            Arguments:
                name: The name of the customer
                email: The email of the customer
                phone: The phone no of the customer

            Returns:
                The customer object of the new customer
            
            Raises:
                Exception: Any unexpected errors
        """
        try:
            with db.get_session() as session:
                new_customer = Customer(name=name, email=email, phone=phone)
                session.add(new_customer)
                session.commit()
                session.refresh(new_customer)
                return new_customer
        except Exception as e:
            print("An unexpected error occured:",e)

    @staticmethod
    def get_customer_details(customer_id: int=None):
        """
            Retrieves either one customer detail or all the customer 
            details based on whether the customer_id is provided or not.

            Arguments:
                customer_id: The ID of the customer

            Returns:
                Retrieves the customer query object of either one customer or
                all the customers
            
            Raises:
                Exception: Any unexpected errors
        """
        try:
            with db.get_session() as session:
                if customer_id:
                    return session.query(Customer).filter(Customer.id == customer_id).first()
                return session.query(Customer).all()
        except Exception as e:
            print("An unexpected error occured:",e)


    @staticmethod
    def update_customer(customer_id: int, name: str = None, email: str = None, phone: str = None, loyalty_level: str = None):
        """
            Updates the customer details

            Arguments:
                customer_id: The ID of the customer
                name: The name of the customer
                email: The email of the customer
                phone: The phone of the customer

            Returns:
                The updated customer object
            
            Raises:
                Exception: Any unexpected errors
        """
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

    @staticmethod
    def delete_customer(customer_id: int):
        """
            Deletes the customer details

            Arguments:
                customer_id: The ID of the customer

            Returns:
                False: if the customer is not found
                True: if the customer is deleted
            
            Raises:
                Exception: Any unexpected errors
        """
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

    @staticmethod
    def fetch_all_customers_with_total_spending():
        """
            Gets all the customer and their total spending details in the 
            descending order of the total amount spent

            Arguments:
                None

            Returns:
                Query object of the all the customers details in the 
                descending order of their total spending
            
            Raises:
                Exception: Any unexpected errors
        """
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


    @staticmethod
    def compute_total_spending():
        """
            Gets all the customer IDs and their total spending details 

            Arguments:
                None

            Returns:
                Query object of the all the customers details in the 
                descending order of their total spending
            
            Raises:
                Exception: Any unexpected errors
        """
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
