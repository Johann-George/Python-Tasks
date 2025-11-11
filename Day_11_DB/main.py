import pandas as pd
from pydantic import ValidationError
from crud.order import OrderService
from crud.customer import CustomerService
from schemas.schema import CustomerInput, OrderInput
from model.models import LoyaltyLevel

def add_customer_with_orders():
    """
        Stores the customer data along with their orders to the database

        Arguments: 
            None

        Returns: 
            None

        Raises:
            ValidationError: 
                Any errors in the value the customer inputs
            ValueError:
                Any input other than a digit
    """
    try:
        name = input("Enter your name:")
        email = input("Enter your email:")
        phone = input("Enter your phone no:")
        customer_data = CustomerInput(name=name, email=email, phone=phone)
        new_customer = CustomerService.create_customer(customer_data.name, customer_data.email, customer_data.phone)

    except ValidationError as e:
        print("Invalid customer data")
        print(e)
        return

    while True:
        try:
            no_of_orders = int(input("Enter the number of different items ordered:"))
            if no_of_orders<=0:
                raise ValueError
            break
        except ValueError:
            print("Please enter a valid positive integer")

    for i in range(no_of_orders):
        try:
            item_name = input("Enter the name of the item:")
            item_price = int(input("Enter the price of the item:"))
            item_quantity = int(input("Enter the quantity:"))
            order_data = OrderInput(item_name=item_name, item_price=item_price, item_quantity=item_quantity)
            OrderService.create_order(order_data.item_name, order_data.item_price, order_data.item_quantity, new_customer.id)
        except ValidationError as e:
            print("Invalid data order")
            print(e)
        except ValueError:
            print("Please enter valid numeric values for price and quantity")


def calculate_total_value(order_id: int):
    """
        Computes the total order amount based on the updated value of 
        price or quantity

        Arguments: 
            order_id: The ID of the order

        Returns: 
            The total amount after updation

        Raises:
            ValueError:
                Any input other than a digit
            Exception:
                Any unexpected errors
    """
    try:
        order = OrderService.get_order_by_id(order_id)
        update_price = int(input("Do you want to update price (0/1):"))
        update_quantity = int(input("Do you want to update quantity (0/1):"))
        if update_price:
            updated_price = float(input("Enter the updated price:"))
            OrderService.update_order(order.id, price=updated_price)
        if update_quantity:
            updated_quantity = int(input("Enter the updated quantity:"))
            OrderService.update_order(order.id, quantity=updated_quantity)
        updated_order = OrderService.get_order_by_id(order_id) 
        return updated_order.price * updated_order.quantity
    except ValueError:
        print("The input must contain only digits")
    except Exception as e:
        print("An unexpected error occured in calculate_total_value():",e)

def fetch_all_customer_and_spendings():
    """
        Fetches all the customer details and thier total spending
        in the descending order

        Arguments: 
            None

        Returns: 
            The dataframe containing the customer details

        Raises:
            ValueError:
                Any input other than a digit
            Exception:
                Any unexpected errors
    """
    try:
        results = CustomerService.fetch_all_customers_with_total_spending()
        df = pd.DataFrame(results, columns=["Customer ID", "Customer Name", "Email", "Phone", "Total Spending"])
        return df
    except Exception as e:
        print("An unexpected error occured fetch_all_customer_and_spendings():", e)

def update_loyalty_level():
    """
        Computes the total spending of each customer and updates their
        loyalty level

        Arguments: 
            None

        Returns: 
            None

        Raises:
            Exception:
                Any unexpected errors
    """
    try:
        spending_data = CustomerService.compute_total_spending()

        for customer_id, total_spending in spending_data:
            if total_spending > 10000:
                level = LoyaltyLevel.GOLD
            elif total_spending >= 5000:
                level = LoyaltyLevel.SILVER
            else:
                level = LoyaltyLevel.BRONZE

            CustomerService.update_customer(customer_id=customer_id, loyalty_level=level) 
    except Exception as e:
        print("An unexpected error occurred update_loyalty_level():", e)


if __name__ == "__main__":
    
    try:
        add_customer_with_orders()
        order_id = int(input("Enter the order ID:"))
        total_value = calculate_total_value(order_id=order_id)
        print("Total Order value:", total_value)
        df = fetch_all_customer_and_spendings()
        print(df)
        update_loyalty_level()

    except Exception as e:
        print(f"Error: {e}")
