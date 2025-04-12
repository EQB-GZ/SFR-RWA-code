import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime
from sfr00_param_controls import sql_yr, sql_time, month_start, dti_raw

myServer = "EQDWP01"
myServer_risk='EQSQLT01\Risk'
myDriver = "{ODBC Driver 17 for SQL Server}" ##Chenxi Li 2025-04-02
myDB = "ET_Finance_Production"
myDB_risk = 'Risk_Analytics'
user = "EQUITAD\cli"


cnxn = pyodbc.connect(
    server=myServer,
    database=myDB,
    user=user,
    password="",
    driver=myDriver,
    Trusted_connection="yes",
    MARS_connection="yes"
)

cnxn_risk = pyodbc.connect(
    server=myServer_risk,
    database=myDB_risk,
    user=user,
    password="",
    driver=myDriver,
    Trusted_connection="yes",
    MARS_connection="yes"
)

#query_ur = f"""SELECT *
#  FROM [ET_Finance_Production].[dbo].[tb_X_Risk_Model_Coefficient]
#  where RiskModelDescription in ('X7 Unemployment Rate PD Final Value','X8 Unemployment Rate LGD Final Value')
#  and ValidFromDate<='{dti_raw}'and ValidToDate>'{dti_raw}'
#  """

#dt_ur_raw = pd.read_sql_query(query_ur, cnxn)
#PD_macro = dt_ur_raw.loc[
#    dt_ur_raw.RiskModelDescription == 'X7 Unemployment Rate PD Final Value', ['Coefficient']].squeeze()
#macro_LGD = dt_ur_raw.loc[
#    dt_ur_raw.RiskModelDescription == 'X8 Unemployment Rate LGD Final Value', ['Coefficient']].squeeze()
##Chenxi Li 2025-04-02
month_end = str(dti_raw)
query_runid = f"""SELECT max(FeedID) as RunID
  FROM [ET_Finance_Production].[dbo].[tb_RE_log]
  where Reporting_date='{month_end}'"""
dt_runid = pd.read_sql_query(query_runid, cnxn)
runid = dt_runid.loc[:, ['RunID']].squeeze()

query_woe_pd = f"""SELECT *
  FROM [ET_Finance_Production].[dbo].[tb_X_WoE_PD_Result]
  where RunID={runid}
  """
dt_woe_pd = pd.read_sql_query(query_woe_pd, cnxn)

query_woe_lgd = f"""SELECT *
  FROM [ET_Finance_Production].[dbo].[tb_X_WoE_LGD_Result]
  where RunID={runid}
  """
dt_woe_lgd = pd.read_sql_query(query_woe_lgd, cnxn)

#query_post_crm = f"""SELECT *
#  FROM [ET_Finance_Production].[dbo].[tb_Post_CRM_Output_G2]
#  where RunID={runid} """
#dt_post_crm = pd.read_sql_query(query_post_crm, cnxn)

#dt_results = pd.merge(dt_post_crm,
#                      pd.merge(dt_woe_pd, dt_woe_lgd, on='LoanNumber', how='outer', suffixes=['_PD', '_LGD']),
#                      on='LoanNumber', how='outer')

#dt_results["Insured_class"] = np.where(dt_results.Final_Post_CRM_PD == 0.00001, 'CMHC',
#                                       np.where(dt_results.Final_Post_CRM_PD.isna(), 'PMI', 'Uninsured'))

#query_CAR=f"""select Loan_Number,Insurer FROM [ET_Finance_Production].[dbo].[CAR23_PMI_Insured]
#           where End_date='9999-12-31'
#           """
query_insurer=f"""select Loan_Number,Insurer_Name FROM [Risk_Analytics].[dbo].[Risk_Consolidated_RESL] 
           where Year='{sql_yr}' and Time='{sql_time}'
           """
dt_insurer = pd.read_sql_query(query_insurer, cnxn_risk)
dt_insurer["Insured_class"] = np.where((dt_insurer.Insurer_Name == 'Genworth') | (dt_insurer.Insurer_Name == 'Canada Guaranty'), 'PMI', dt_insurer.Insurer_Name)
dt_results = pd.merge(pd.merge(dt_woe_pd, dt_woe_lgd, on='LoanNumber', how='outer', suffixes=['_PD', '_LGD']),dt_insurer,
                      left_on='LoanNumber',right_on='Loan_Number', how='left')
#Chenxi Li 2025-04-02

# FN_query = f"""select Loan_Number,Advance_Amount from [ET_Finance_Production].[dbo].[tblrelational_TB_Third_Party]
# where Year='{sql_yr}' and Time='{sql_time}' """
# PQ_query = f"""select Loan_Number,Advance_Amount from [ET_Finance_Production].[dbo].[tblrelational_TB_Paradigm]
# where Year='{sql_yr}' and Time='{sql_time}' """

retail_query = f""" select r.Loan_Number,r.Advance_Amount,r.CIF_Number
 from [ET_Finance_Production].[dbo].[tblrelational_TB_Retail] as r 
 where r.Year='{sql_yr}' and r.Time='{sql_time}' 
 union 
 select fn.Loan_Number,fn.Advance_Amount,fn.CIF_Number
  from [ET_Finance_Production].[dbo].[tblrelational_TB_Third_Party] as fn 
  where fn.Year='{sql_yr}' and fn.Time='{sql_time}'
union 
select pq.Loan_Number,pq.Advance_Amount,pq.CIF_Number
 from [ET_Finance_Production].[dbo].[tblrelational_TB_Paradigm] as pq
 where pq.Year='{sql_yr}' and pq.Time='{sql_time}' """
df_retail_raw = pd.read_sql_query(retail_query, cnxn)

# retail_query = f"""select [Loan_Number],[CIF_Number],[Advance_Date]
# from dbo.tblrelational_TB_Retail where Year='{sql_yr}' and Time='{sql_time}' """

query_cif=f"""select CIF_Number,Firm_Or_Individ FROM [ET_Finance_Production].[dbo].[tblrelational_CIF_Snapshot] 
           where Year='{sql_yr}' and Time='{sql_time}'
           """
df_cif = pd.read_sql_query(query_cif, cnxn)
df_retail_raw2=pd.merge(df_retail_raw,df_cif, on=['CIF_Number'], how='left')

property_query = f"""select a.*,b.[Property_Class],b.Property_Description,b.Property_Type  
FROM [ET_Finance_Production].[dbo].[tblrelational_Property] a 
left outer join [ET_Finance_Production].[dbo].[tblrelational_Property_Type] b 
on a.Property_Code=b.Property_Code
where a.Year='{sql_yr}' and a.time='{sql_time}'"""
df_prop_raw = pd.read_sql_query(property_query, cnxn)
prop_col = ["Loan_Number", "Annual_Revenue", 'Occupant']
df_prop_raw2 = df_prop_raw.sort_values('Property_Number').groupby('Loan_Number').head(1)

portfolio_query = f"""SELECT [Loan_Number],[EAD],[Remaining_Principal_EQB],[Property_Value],[Loan_Class_Post_CRM]
FROM [ET_Finance_Production].[dbo].[t_SFR_Risk_Rating_Attributes] where Year='{sql_yr}' and time='{sql_time}' """
df_raw1 = pd.read_sql_query(portfolio_query, cnxn)

df_raw2 = pd.merge(df_raw1, df_prop_raw2[prop_col], on=['Loan_Number'], how='left')
df_raw3 = pd.merge(df_raw2, df_retail_raw2, on=['Loan_Number'], how='left')
#df = pd.merge(df_raw3, dt_results, left_on=['Loan_Number'],right_on=['LoanNumber'], how='left')
df = pd.merge(df_raw3, dt_results, on=['Loan_Number'], how='left') #Chenxi Li 2025-04-02

applicant_query = f"""Select * FROM (SELECT year, time, loan_number, CIF_Number, 'Applicant' as 'Borrower_Type' FROM 
[ET_Finance_Production].[dbo].[tblrelational_TB_Retail] where year ='{sql_yr}' and time ='{sql_time}' and 
underwriter_Code in ('cma','prm') union Select c.year, c.time, c.loan_number, c.CIF_Number, 'Co_Applicant' From 
ET_Finance_Production.dbo.tblrelational_Co_Applicant_Snapshot c left outer join 
ET_Finance_Production.dbo.tblrelational_TB_Retail r on r.year = c.year and r.time = c.time and r.loan_number = 
c.loan_Number where r.year ='{sql_yr}' and r.time ='{sql_time}' and r.Underwriter_Code in ('cma','prm')) a left join  
ET_Finance_Production.dbo.tblrelational_Income_Snapshot i on i.CIF_Number = a.CIF_Number and a.year = i.year and 
a.time = i.time """
dt_applicant = pd.read_sql_query(applicant_query, cnxn)

##Chenxi Li 2025-04-02
addition_query = f"""SELECT sfr.Loan_Number,sfr.X1_Average_Beacon_Score_PD,sfr.X2_Current_LTV_PD,
sfr.X2_Loan_Class_PD,sfr.X3_Worst_Delinquency_Two_Years_Max_PD,sfr.X4_Province_PD,
sfr.X1_FSA_LGD,sfr.X1_Province_LGD,X2_LTV_LGD,
sfr.X3_Appraisal_LGD,sfr.X4_Occupancy_LGD,
sfr.Source,sfr.Metro,sfr.Dwelling_Type,

 retail.Advance_Date,
--retail.Advance_Amount,
retail.Loan_Rate,retail.Loan_Rate_New,retail.Maturity_Date,
class.IRB_Asset_Class,
(case when retail.Loan_Rate like '%P%' then 'Variable'
else 'Fixed'
end) as Interest_Type
--retail.Original_Maturity_Date,retail.Original_Principal_Amount,
  FROM [ET_Finance_Production].[dbo].[t_SFR_Risk_Rating_Attributes] as sfr
    left join [ET_Finance_Production].[dbo].[t_Loan_Retail_IRB_Data] as class
   on sfr.Year=class.year and sfr.time=class.time and sfr.Loan_Number=class.Loan_Number
  left join
  
 ( select r.Loan_Number,r.Advance_Date,
 r.Loan_Rate,r.Loan_Rate_New,r.Maturity_Date
 from [ET_Finance_Production].[dbo].[tblrelational_TB_Retail] as r 
 where r.Year='{sql_yr}' and r.Time='{sql_time}' 
 union 
 select fn.Loan_Number,fn.Advance_Date,
 fn.Loan_Rate,fn.Loan_Rate_New,fn.Maturity_Date
  from [ET_Finance_Production].[dbo].[tblrelational_TB_Third_Party] as fn 
  where fn.Year='{sql_yr}' and fn.Time='{sql_time}'
union 
select pq.Loan_Number,pq.Advance_Date,
 pq.Loan_Rate,pq.Loan_Rate_New,pq.Maturity_Date
 from [ET_Finance_Production].[dbo].[tblrelational_TB_Paradigm] as pq
 where pq.Year='{sql_yr}' and pq.Time='{sql_time}' 
  ) retail

  on sfr.Loan_Number=retail.Loan_Number 
  where sfr.Year='{sql_yr}' and sfr.Time='{sql_time}' 
"""
dt_addition = pd.read_sql_query(addition_query, cnxn)
