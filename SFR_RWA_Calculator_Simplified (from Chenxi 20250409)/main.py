import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
#print('SFR PD - EQB (c) 2022 Sep')

col_pre_pd = 'PD'
col_pre_lgd = 'LGD'
mrs_ind = True
# start_time = datetime.now()
# print(str(datetime.now()) + ": Start Loading Data -----------------------")
# Add Rental Flag to data for RWA 2023 calculation

from sfr03_execution_rwa import func_execution_rwa

output_grp = "Insured_class"
df, df_report, summary_tbl, df_corp= func_execution_rwa(col_pre_pd, 'Final_LGD', mrs_ind, 'insured_PMI_ratio', output_grp)

from sfr01_sql_data_input import dt_addition
from sfr00_param_controls import dwelling_ref

#dt_addition['Dwelling_Description'] = dt_addition.Dwelling_Type.map(dwelling_ref)
df['Dwelling_Description'] = df.Dwelling_Type.map(dwelling_ref)
#dt_final = pd.merge(df[['Loan_Number', 'Insured_class', 'Advance_Amount', 'CalibratedPD', 'Final_LGD', 'EAD','Property_Value',
#                        'RiskRating_PD', 'RiskRating_LGD',
#                        'RWA_AIRB', 'RWA_standardized', 'RWA_AIRB_2019', 'RWA_standardized_2019']],
#                    dt_addition, on='Loan_Number', how='left')
# dt_final.to_csv('Residential_202301.csv',index=False)
# summary_tbl.to_csv('SFR_RWA_Summary_202301.csv',index=True)
dt_final = pd.merge(df[['Loan_Number','IRB_Asset_Class', 'Insured_class', 'RWA_AIRB', 'CalibratedPD','RiskRating_PD','Add_on_LGD', 'Final_LGD','RiskRating_LGD', 'EAD','insured_PMI_ratio',
               'Advance_Amount','corr_uninsured','corr_insured','RWA_standardized']],df_corp[['Loan_Number','Years_to_maturity','Maturity_adj']],on='Loan_Number', how='left')


YearTime = str(input("Again, remind what is the year and month you are looking for (e.g. YYYYMM): "))  # 20250410 added by George

# dt_final.to_csv('SFR_202412_v2.csv',index=False)  
# summary_tbl.to_csv('SFR_RWA_Summary_202412.csv',index=True)


dt_final.to_csv('SFR_'+ YearTime + '_v2.csv',index=False)  
summary_tbl.to_csv('SFR_RWA_Summary_'+  YearTime +'.csv',index=True)
