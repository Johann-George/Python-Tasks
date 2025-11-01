import csv
from faker import Faker
import uuid
import random

class Transactions:

    def __init__(self,file):

        self.total_success_transaction = 0
        self.total_success_amount = 0
        self.total_failed_transaction= 0
        self.total_failed_amount = 0
        self.total_pending_transaction = 0
        self.total_pending_amount = 0
        self.highest_transaction = float('-inf')
        self.data_dict = {}
        self.writing_file = file


    def generate_csv_file(self, csv_file):

        """
            Generates 500 records of transaction data and stores it in transactions.csv

            Arguments:
                csv_file: The file which contains the transaction details
            
            Returns:
                No return statements
            
            Raises:
                Exception: Any other unexpected errors
        """

        try:
            fake = Faker()
            status_values = ['Success', 'Failed', 'Pending']
            field_names = ['Transaction_id','customer_name','Date','Amount','Status']
            with open(csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(field_names)

                for _ in range(500):
                    self.transaction_id = uuid.uuid4()
                    self.customer_name = fake.name()
                    self.date = fake.date()
                    self.amount = random.randint(100,5000)
                    self.status = random.choice(status_values)
                    writer.writerow([self.transaction_id, self.customer_name, self.date, self.amount, self.status])
        except Exception as e:
            print("An unexpected error occured generate_csv_file():", e)

    def save_in_dictionary(self, csv_file):

        """
            Save the csv file as key value pair in a dictionary

            Arguments:
                csv_file: The file which contains the transaction data
            
            Returns:
                No return statements
            
            Raises:
                FileNotFoundError: If the csv file is not found: save_in_dictionary()
                Exception: Any other unexpected errors: save_in_dictionary()
        """
        try:
            with open(csv_file,'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    for key, value in row.items():
                        self.data_dict.setdefault(key, []).append(value)
        except FileNotFoundError:
            print("The csv file is not found: save_in_dictionary()")
        except Exception as e:
            print("An unexpected error occured: save_in_dictionary()",e)


    def total_transaction_and_avg_amount_per_status(self):

        """
            Calculates the total transaction per status and average amount per status

            Arguments:
                No arguments
            
            Returns:
                No return statements
            
            Raises:
                FileNotFoundError: If the csv file is not found: save_in_dictionary()
                Exception: Any other unexpected errors: save_in_dictionary()
        """
        for index, status in enumerate(self.data_dict['Status']):
            if status == "Success":
                self.total_success_transaction += 1
                self.total_success_amount += int(self.data_dict['Amount'][index])
            elif status == "Pending":
                self.total_pending_transaction += 1
                self.total_pending_amount += int(self.data_dict['Amount'][index])
            else:
                self.total_failed_transaction += 1
                self.total_failed_amount += int(self.data_dict['Amount'][index])
        self.avg_success_transaction = self.total_success_amount/self.total_success_transaction
        self.avg_pending_transaction = self.total_pending_amount/self.total_pending_transaction
        self.avg_failed_transaction = self.total_failed_amount/self.total_failed_transaction

    def highest_transaction_and_customer(self):
        
        """
            Finding the highest transaction, its transaction id and customer name

            Arguments:
                No arguments
            
            Returns:
                No return statements
            
            Raises:
                Exception: Any other unexpected errors: save_in_dictionary()
        """

        try:
            for amount in self.data_dict['Amount']:
                if (int(amount) > self.highest_transaction):
                    self.highest_transaction = int(amount)
        except Exception as e:
            print("An unexpected error occured: highest_transaction_and_customer()",e)
    

    def save_summary(self, txt_file, pdf_file):

        """
            Saves the summary of the data to a txt and pdf file

            Arguments:
                txt_file: The text file to which the summary is written
                pdf_file: The pdf file to which the summary is written
            
            Returns:
                No return statements
            
            Raises:
                Exception: Any unexpected errors
        """

        try:
            file = open(txt_file,"w") 
            file.write(f"Total Transactions: 500 Success:{self.total_success_transaction} Failed: {self.total_failed_transaction} Pending: {self.total_pending_transaction} Average (Success): {self.avg_success_transaction} Highest Transaction : {self.transaction_id} by {self.customer_name} - {self.highest_transaction}")
        
            file = open(pdf_file,"w") 
            file.write(f"Total Transactions: 500 Success:{self.total_success_transaction} Failed: {self.total_failed_transaction} Pending: {self.total_pending_transaction} Average (Success): {self.avg_success_transaction} Highest Transaction : {self.transaction_id} by {self.customer_name} - {self.highest_transaction}")
        except Exception as e:
            print("An unexpected error occured:", e)

c = Transactions()
c.generate_csv_file('transactions.csv')
c.save_in_dictionary('transactions.csv')
c.total_transaction_and_avg_amount_per_status()
c.highest_transaction_and_customer()
c.save_summary("summary.txt","summary.pdf")

# def main()
    
    # class calling + classs def calling

# wri = 
