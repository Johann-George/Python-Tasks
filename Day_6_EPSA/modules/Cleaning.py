import pandas as pd
import re

class Cleaning:

    def __init__(self, excel_file):
        self.all_sheets = pd.read_excel(excel_file,sheet_name=None)
        self.employee_df = self.all_sheets['employees']
        self.salaries_df = self.all_sheets['salaries']
        self.performance_df = self.all_sheets['performance']

    def get_all_sheets(self):
        return self.all_sheets

    def display_data(self):
        print(self.employee_df.head())
        print(self.salaries_df.head())
        print(self.performance_df.head())

    def convert_join_date_to_date_format(self):
        try:
            self.employee_df['Join_Date'] = pd.to_datetime(self.employee_df['Join_Date'], format='mixed')
            self.employee_df['Join_Date'] = self.employee_df['Join_Date'].dt.strftime('%d-%m-%Y')
        except Exception as e:
            print("An unexpected error occured:",e)        


    def drop_duplicate_emp_ids_and_check_uppercase(self):
        try:
            is_upper = self.employee_df['Emp_ID'].str.isupper()
            print("Is emp id always upper:", is_upper)
            self.employee_df.drop_duplicates(subset=['Emp_ID'], inplace=True)
        except Exception as e:
            print("An unexpected error occured:",e)        


    def clean_string(self, text):
        try:
            if isinstance(text, str):
                text = text.strip()
                text = re.sub(r'\s+', ' ', text)
            return text
        except Exception as e:
            print("An unexpected error occured:",e)        


    def cleaned_dataframe(self):
        try:
            self.employee_df = self.employee_df.applymap(self.clean_string)
            self.salaries_df = self.salaries_df.applymap(self.clean_string)
            self.performance_df = self.performance_df.applymap(self.clean_string)
        except Exception as e:
            print("An unexpected error occured:",e)        

