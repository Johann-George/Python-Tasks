import pandas as pd
from functools import reduce

class Analysis:

    def calculate_avg_monthly_net_salary_per_dept(self):
        try:
            net_salary_df = pd.read_excel('employee_output_analysis.xlsx', sheet_name='Sheet1')
            avg_salary = net_salary_df.groupby('Department', as_index=False)['Net Salary'].mean()
            with pd.ExcelWriter('employee_output_analysis.xlsx', engine='openpyxl', mode='a') as writer:
                avg_salary.to_excel(writer, sheet_name='Analysis Results', index=False)
        except Exception as e:
            print("An unexpected error occured:",e)        


    def identify_underpaid_top_employees(self):
        try:
            df = pd.read_excel('employee_output_analysis.xlsx', sheet_name='Sheet1')
            dept_avg = df.groupby('Department', as_index=False)['Net Salary'].mean()
            dept_avg.rename(columns={'Net Salary': 'Dept_Avg_Salary'}, inplace=True)
            df = df.merge(dept_avg, on='Department', how='left')
            underpaid_top = df[(df['Performance_Rating'] > 4.5) & (df['Net Salary'] < df['Dept_Avg_Salary'])]
            underpaid_name = ', '.join(underpaid_top['Full_Name'].tolist())
            summary_df = pd.DataFrame({'Underpaid Top Performers': [underpaid_name]})
            with pd.ExcelWriter('employee_output_analysis.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                summary_df.to_excel(writer, sheet_name='Analysis Results', index=False)
        except Exception as e:
            print("An unexpected error occured:",e)        

        
    def create_new_sheet(self):
        try:

            df = pd.read_excel('employee_output_analysis.xlsx', sheet_name='Sheet1')
            summary_df = (
                df.groupby(['Department', 'Quarter'], as_index=False)
                .agg({'Performance_Rating': 'mean', 'Net Salary': 'mean'})
                .rename(columns={
                    'Performance_Rating': 'Avg_Rating',
                    'Net Salary': 'Net_Salary_per_dept'
                })
            )
            with pd.ExcelWriter('employee_output_analysis.xlsx', engine='openpyxl', mode='a') as writer:
                summary_df.to_excel(writer, sheet_name='Dept_Quarter_Summary', index=False)
        except Exception as e:
            print("An unexpected error occured:",e)        


    def udpate_sheet(self, all_sheets):
        try:
            df = pd.read_excel('employee_output_analysis.xlsx', sheet_name='Sheet1')
            df = df.drop('Quarter', axis=1)
            mean_ratings = df.groupby('Emp_ID', as_index=False)['Performance_Rating'].mean()
            employee_info = df.drop_duplicates(subset=['Emp_ID']).drop(columns=['Performance_Rating'])
            final_df = pd.merge(employee_info, mean_ratings, on='Emp_ID', how='left')
            print(final_df)
            # employees_df = all_sheets['employees']
            # salaries_df = all_sheets['salaries']
            # performance_df = all_sheets['performance']
            
            # performance_mean = performance_df.groupby(['Emp_ID'], as_index=False)['Performance_Rating'].mean()

            # dfs = [employees_df, salaries_df, performance_mean]
            # self.merged_df = reduce(lambda left, right: pd.merge(left, right, on='Emp_ID', how='left'), dfs)
            with pd.ExcelWriter('employee_output_analysis.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                final_df.to_excel(writer, sheet_name='Sheet1', index=False)
        except Exception as e:
            print("An unexpected error occured:",e)        

