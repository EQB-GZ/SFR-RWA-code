from sfr01_sql_data_input import df, dt_applicant,dt_addition
from sfr00_param_controls import occupancy_ref, rental_ref, rental_income_ref
import pandas as pd
import numpy as np
import myFunctions

# Uniforming the 'Advanced and renewal' dates. They will be required for grandfathering rules.
# df.loc[:,'Advanced_Date_clnd'] = df.Advance_Date.apply(lambda s: int(s[:4]+ s[5:7]))
# df.loc[:, 'Advanced_Date_clnd'] = df.Advance_Date.apply(lambda s: int(str(s.year) + str('{:02d}'.format(s.month))))
#
# df.loc[:, 'Renewal_Date_clnd'] = df.loc[:, 'Renewal_Date'].apply(lambda s: myFunctions.Renewal_Date_cleaner(s))
# Calculating Current LTV value
df['Current_LTV'] = df.Remaining_Principal_EQB * 100 / df.Property_Value

# The insured status of loans. It is required for grandfathering in Addon Calculation and later for RWA calcultions.

# insured_dict = {'2': 'CMHC', 'A': "Sagen", "C": "CG",
#                 "1": "Uninsured", "8": "Uninsured", "4": "Uninsured",
#                 'CMHC': 'CMHC', "GE": "Sagen", "CG": "CG",
#                 "UNINSURED": "Uninsured"}
# df["Loan_Class_Post_CRM"] = np.where()
df.loc[:, "Loan_Class_Post_CRM_unified"] = df.loc[:, "Loan_Class_Post_CRM"].apply(lambda s: str(s))
df.loc[:, "Loan_Class_Post_CRM_unified"] = df.loc[:, "Loan_Class_Post_CRM_unified"].str.upper()
# df.loc[:, "Insured_class"] = df.Loan_Class_Post_CRM_unified.map(insured_dict)
df["Insured_class"] = np.where(df.Loan_Class_Post_CRM_unified == 'A', 'Sagen',
                               np.where(df.Loan_Class_Post_CRM_unified == 'C', 'CG',
                                        np.where(df.Loan_Class_Post_CRM_unified.isin(['F','2']) , 'CMHC', 'Uninsured')))

df.Occupant = df.Occupant.str.upper()
df['Occupancy_Type'] = df.Occupant.map(occupancy_ref)
df['Owner_Occupied'] = df.Occupant.map(rental_ref)

df.loc[:, 'Rental_Class'] = df.loc[:, 'Occupant'].map(rental_income_ref)
cond_rental = df.Rental_Class.isna()
df.loc[cond_rental, 'Rental_Class'] = 'N'
df.loc[:, 'Rental_Class_Occupant'] = df.loc[:, 'Occupant'].map(rental_income_ref)
cond_rental = df.Rental_Class_Occupant.isna()
df.loc[cond_rental, 'Rental_Class_Occupant'] = 'N'

#dt_applicant['Annual_Gross'] = dt_applicant['Annual_Gross'].astype(float)
dt_applicant['Annual_Gross'] = pd.to_numeric(dt_applicant['Annual_Gross'], errors="coerce").isnull() #Chenxi 2025-04-02
dt_applicant['Annual_Gross'] = dt_applicant['Annual_Gross'].fillna(0)
dt_applicant['Rental_Gross'] = 0.0
# dt_applicant['Rental_Gross'] = dt_applicant['Rental_Gross'].astype(float)
# Chenxi 2022-04-26 #cond add at right of equation
dt_applicant.loc[dt_applicant['Income_Type_Code'] == 5, 'Rental_Gross'] = dt_applicant.loc[
    dt_applicant['Income_Type_Code'] == 5, 'Annual_Gross']
dt_loan_income = dt_applicant.groupby(['loan_number']).agg(Total_income_amount=('Annual_Gross', 'sum'),
                                                           Rental_income_amount=('Rental_Gross', 'sum')
                                                           )

# cond_rental_income
dt_loan_income['Loan_Number'] = dt_loan_income.index

df = pd.merge(df, dt_loan_income, how='left', on='Loan_Number')
# print(df.shape)
df['Annual_Revenue'] = df['Annual_Revenue'].astype(float)
df.Annual_Revenue = df.Annual_Revenue.fillna(0)
# df['rental_ratio']=np.nan


# df['rental_ratio']= df.Rental_income_amount/df.Total_income_amount
df.Rental_income_amount = df.Rental_income_amount.fillna(0)
df.Total_income_amount = df.Total_income_amount.fillna(0)
df["rental_ratio"] = np.where((df.Occupant == "RI"),
                              np.where(df.Total_income_amount + df.Annual_Revenue == 0,0,
                                       df.Annual_Revenue / (df.Total_income_amount + df.Annual_Revenue)),
                              np.where(df.Total_income_amount == 0,0,df.Rental_income_amount / df.Total_income_amount))  # Chenxi 2025-04-02

# create Rental Ratio Bucket for Analysis #2022-04-21
# cond_rental_income= (df.loc[:,'Rental_Class']=='Y')
cond_rental_income = (df.Rental_Class == 'Y') & (df.rental_ratio > 0.5)

# df['Rental_Income'] ='N'
# df.loc[cond_rental_income,'Rental_Income'] ='Y'
df['Rental_Income'] = np.where(cond_rental_income, 'Y', 'N')

df["Rental_Ratio_Bucket"] = "No Rental Income"
# df.loc[df.Rental_Class_Occupant=="Y","Rental_Ratio_Bucket"] ='Rental'

cond_rental_ratio = ((0 < df.loc[:, 'rental_ratio']) & (df.loc[:, 'rental_ratio'] <= 0.1))
df.loc[cond_rental_ratio, "Rental_Ratio_Bucket"] = "Rental Income 0 - 10%"

cond_rental_ratio = ((0.1 < df.loc[:, 'rental_ratio']) & (df.loc[:, 'rental_ratio'] <= 0.2))
df.loc[cond_rental_ratio, "Rental_Ratio_Bucket"] = "Rental Income 10% - 20%"

cond_rental_ratio = ((0.2 < df.loc[:, 'rental_ratio']) & (df.loc[:, 'rental_ratio'] <= 0.3))
df.loc[cond_rental_ratio, "Rental_Ratio_Bucket"] = "Rental Income 20% - 30%"

cond_rental_ratio = ((0.3 < df.loc[:, 'rental_ratio']) & (df.loc[:, 'rental_ratio'] <= 0.4))
df.loc[cond_rental_ratio, "Rental_Ratio_Bucket"] = "Rental Income 30% - 40%"

cond_rental_ratio = ((0.4 < df.loc[:, 'rental_ratio']) & (df.loc[:, 'rental_ratio'] <= 0.5))
df.loc[cond_rental_ratio, "Rental_Ratio_Bucket"] = "Rental Income 40% - 50%"

cond_rental_ratio = ((df.loc[:, 'rental_ratio'] > 0.5))
df.loc[cond_rental_ratio, "Rental_Ratio_Bucket"] = "Rental Income 50% up"

# add uninsured deductible sub-exposure ratio for PMI RWA calculation
df["Advance_Amount"] = df["Advance_Amount"].astype(float)
df.Advance_Amount = df.Advance_Amount.replace('NaN', np.nan).fillna(0)

# df["insured_PMI_ratio"]=(df.Remaining_Principal_Post_CRM-0.1*df.Advance_Amount)/df.Advance_Amount
df["deductible_amount"] = 0.1 * df.Advance_Amount
df["insured_PMI_ratio"] = np.where(df.EAD > 0, (df.EAD - df.deductible_amount) / df.EAD, 0)


df = pd.merge(df,dt_addition, on='Loan_Number', how='left')

