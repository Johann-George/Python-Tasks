import pandas as pd
from functools import reduce

class Transformation:

    def merge_all_sheets(self, all_sheets):
        try:
            employees_df = all_sheets['employees']
            salaries_df = all_sheets['salaries']
            performance_df = all_sheets['performance']
            
            performance_mean = performance_df.groupby(['Emp_ID', 'Quarter'], as_index=False)['Performance_Rating'].mean()

            dfs = [employees_df, salaries_df, performance_mean]
            self.merged_df = reduce(lambda left, right: pd.merge(left, right, on='Emp_ID', how='left'), dfs)
        except Exception as e:
            print("An unexpected error occured", e)

    def display_data(self):
        print(self.merged_df)

    def add_new_columns(self):
        try:
            split_names = self.merged_df['Full_Name'].str.extract(r'(\w+)\s+([\w\s]+)\s+(\w+)')
            split_names.columns = ['Prefix', 'Given Name', 'Surname']
            self.merged_df = pd.concat([split_names, self.merged_df], axis=1)

            split_address = self.merged_df['Address'].str.extract(r'(\d+)\s+([\w\s]+),\s*(\w+)')
            split_address.columns = ['Street Number', 'Street Address', 'Country']
            self.merged_df = pd.concat([split_address, self.merged_df], axis=1)

            self.merged_df['Net Salary'] = (
                self.merged_df[['Base_Salary', 'Bonus', 'Deductions']]
                .fillna(0)
                .eval('Base_Salary + Bonus - Deductions')
            )
        except Exception as e:
            print("An unexpected error occured", e)


    def export_transformed_data(self):
        try:
            self.merged_df.to_excel('employee_output_analysis.xlsx', index=False, sheet_name='Sheet1')
        except Exception as e:
            print("An unexpected error occured:",e)        

