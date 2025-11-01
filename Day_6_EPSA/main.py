# Day 7

from modules.Cleaning import Cleaning
from modules.Transformation import Transformation
from modules.Analysis import Analysis

def pipeline():
    c = Cleaning('employee_data.xlsx')
    all_sheets = c.get_all_sheets()
    c.display_data()
    c.convert_join_date_to_date_format()
    c.display_data()
    c.drop_duplicate_emp_ids_and_check_uppercase()
    c.display_data()
    c.cleaned_dataframe()
    c.display_data()

    t = Transformation()
    t.merge_all_sheets(all_sheets)
    t.display_data()
    t.add_new_columns()
    t.display_data()
    t.export_transformed_data()

    a = Analysis()
    a.calculate_avg_monthly_net_salary_per_dept()
    a.identify_underpaid_top_employees()
    a.create_new_sheet()
    a.udpate_sheet(all_sheets)

if __name__ == "__main__":
    pipeline()