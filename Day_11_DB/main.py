from sqlalchemy import func
import pandas as pd
from pydantic import ValidationError
from model.models import Customer, Order
from crud.order import update_order, create_order, get_order_by_id
from crud.customer import create_customer, fetch_all_customers_with_total_spending, compute_total_spending, update_customer
from schemas.schema import CustomerInput, OrderInput

def add_customer_with_orders():

    try:
        name = input("Enter your name:")
        email = input("Enter your email:")
        phone = input("Enter your phone no:")
        # if not all([name,email,phone]):
        #     print("Error in getting data")
        #     return False
        customer_data = CustomerInput(name, email, phone)
        new_customer = create_customer(customer_data.name, customer_data.email, customer_data.phone)

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
            create_order(order_data.item_name, order_data.item_price, order_data.item_quantity, new_customer.id)
        except ValidationError as e:
            print("Invalid data order")
            print(e)
        except ValueError:
            print("Please enter valid numeric values for price and quantity")


def calculate_total_value(order_id: int):
    order = get_order_by_id(order_id)
    update_price = int(input("Do you want to update price (0/1):"))
    update_quantity = int(input("Do you want to update quantity (0/1):"))
    if update_price:
        updated_price = float(input("Enter the updated price:"))
        update_order(order.id, price=updated_price)
    if update_quantity:
        updated_quantity = int(input("Enter the updated quantity:"))
        update_order(order.id, quantity=updated_quantity)
    updated_order = get_order_by_id(order_id) 
    return updated_order.price * updated_order.quantity

def fetch_all_customer_and_spendings():
    results = fetch_all_customers_with_total_spending()
    df = pd.DataFrame(results, columns=["Customer ID", "Customer Name", "Email", "Phone", "Total Spending"])
    return df

def update_loyalty_level():
    spending_data = compute_total_spending()

    for customer_id, total_spending in spending_data:
        if total_spending > 10000:
            level = "Gold"
        elif total_spending >= 5000:
            level = "Silver"
        else:
            level = "Bronze"

        update_customer(customer_id=customer_id, loyalty_level=level) 


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

    # finally:
    # session.close()
    #     print("Session closed")