#%%  import packages

import os
import pyodbc 
import pandas as pd
import numpy as np
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import time
import datetime
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import math
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import pickle
import collections
import re
import random
import matplotlib.ticker as mtick
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from scipy import stats
from statsmodels.stats.weightstats import ztest
import copy
import multiprocessing
from joblib import Parallel, delayed
from scipy.stats import kstest

from tqdm import tqdm
import collections
import logging

logging.basicConfig(level=logging.INFO)

#%%################################################################################################################
#                                     SETTTINGS
###################################################################################################################
#date

# take user input so that different users can run at the same time for different FN_Inclusion values

# FN_Inclusion = input('Do you want to include First National? Y or N \n')   #whether to include First National in the Modeling Development; can only be 'Y' or 'N'

run_ID = 11

FN_Inclusion = 'Y' #!!!!!!DO NOT CHANGE without EVERYONE KNOWING IT -  it impacts every code that will be run and can be mistakenly ran/modified be others in the meantime

Data_Collection_Start_Date = '2007-01-01'

Data_Collection_End_Date = '2021-12-31'

One_Day_bf_Covid_Affect_Date = '2019-03-31'  

Covid_Affect_Start_Date  = '2019-04-01' # The covid period is from Apr 2020 to Dec 2020 when deferral treatment in ends. However, due to 1 - year performance window for PD modeling, The period will shift 1 year before

Covid_Affect_End_Date = '2020-12-31'  # same as Bennington PD. Covid Ending date is taken as Dec 2020, when the widest support (CREB) ended already.
   
Default_IS_End_Date  = '2020-12-31'

Default_OOT_Start_Date = '2021-01-01' 

num_cores = multiprocessing.cpu_count() - 6

today = datetime.date.today()

#minimum bin count for WOE transformation
min_woe_bin_count = 100 

# minimum WOE difference used for grouping WOE bins 
min_woe_diff = 0.3

#maxmimum number of bins
max_woe_bins = 10

# Correlation threshold used for multivariate analysis to get clusters of cloumns among which the correlation is lower than threshold
high_corr_theshold=0.5

#MRS segmentation bin
max_mrs_bins = 5

#MRS segementation decision tree
min_mrs_bin_dist = 0.1


#whether to combine early and late delinquent accounts into one segment, default is two segments
combine_earlY_and_late_dlq = True


gini_wgt_oos=2/3
gini_wgt_trn=1/3
gini_wgt_oot=0

#column for Loan Number
Col_Loan_Nbr = 'Loan_Number'


#column for target variable
Col_Target = 'DF_Ind'

#column for snap date variable
col_snap_date =  'SL_Date'

#  skipped drivers
cols_skip = ['Year', 'Twelve_Month_FWD_Default_Ind', 'Underwriter', 'Time', 'SL_YYYYMM', 'SL_Date', 'Run_ID', 'Renew_Date', 'Post_Default_Ind', 'Partner_Portion_Inception', 'Original_LTV_Including_Partner', 'Original_LTV_Excluding_Partner', 'Occupancy_Code', 'Min_Default_Year', 'Min_Default_Date', 'Max_Default_Date', 'Maturity_Date', Col_Loan_Nbr, 'Last_Update_Date', 'Gemstone', 'Funding_TDS_Ratio', 'Funding_GDS_Ratio', 'Funded_Date', 'FSA', 'Default_Ind', 'Comments', 'Combo_Funding_TDS', 'Combo_Funding_GDS', 'City', 'BNI_Max_Guar_Incept', 'BNI_Max_CoApp_Incept', 'BNI_Max_App_CoApp_Incept', 'BNI_Max_All_Incept', 'BNI_App_Incept', 'BFS_Salary_Ind_Orig', 'Beacon_Max_Guar_Incept', 'Beacon_Max_CoApp_Incept', 'Beacon_Max_App_CoApp_Incept', 'Beacon_Max_All_Incept', 'Beacon_App_Incept', 'Arrears_Days', 'AppraisalValue_Inception', 'AdvancedAmt_Incl_Part', 'AdvancedAmt_Excl_Part', 'AdvancedAmount_Total_Exp', 'AdvancedAmount_EQB_Exp','DF_Ind','Number_Of_Loan_Associated_To_Main_CIF_MinSL','Arrears_Status','Gemstone_Rank','Gemstone','LTV_Excl_Part_MD_3M_Chg']

dlq_folder_full_list = ['Dlq_Cur','Dlq_Early','Dlq_Late','Dlq_All']

Covid_Inclusion_InSample_full_list = ['Y', 'N']

Segment_full_list = ['Portfolio','Alt','Prime','Port_Excl_FN','Debug']



#%%################################################################################################################
#                                     Inputs
###################################################################################################################


# Output Driectory

Dir_Root = os.getcwd()

Dir_Results = Dir_Root + '\\'+'Results'

# Dir_Test = os.getcwd()+'\\'+'Test'

Dir_Inputs = Dir_Root + '\\' + 'Inputs'

Dir_Figures = Dir_Root + '\\' + 'Results\Figures'

if FN_Inclusion == 'N':
    Root_Dir_for_Model =  Dir_Results + '\\Excl_FN'

if FN_Inclusion == 'Y':
    Root_Dir_for_Model =  Dir_Results + '\\Incl_FN'


def dir_for_model_func(Root_Dir= Root_Dir_for_Model, Covid_Inclusion_InSample = 'Y', dlq_folder = 'Dlq_Cur', Segment = 'Portfolio'):
    # Folder for model results

    if Covid_Inclusion_InSample == 'Y':
        Sample_Folder = 'Covid Included'    
    else: #Covid started in Jan 2020, but to leave 1 year performance window
        Sample_Folder = 'Covid Excluded'       
    
    Dir_Model = Root_Dir + '\\' + f'{dlq_folder}'+ '\\'+f'{Sample_Folder}'+'\\'+ f'{Segment}'

    return Dir_Model

def Is_End_Date(Covid_Inclusion_InSample):
    if Covid_Inclusion_InSample =='Y':
        IS_End_Date = Default_IS_End_Date
    else:
        IS_End_Date = One_Day_bf_Covid_Affect_Date
    return IS_End_Date    

def OOT_Start_Date(Covid_Inclusion_InSample):
    if Covid_Inclusion_InSample =='Y':
        OOT_Start_Date = Default_OOT_Start_Date 
    else:
        OOT_Start_Date = Covid_Affect_Start_Date

    return OOT_Start_Date

#  Data Files

file_data_types = Dir_Inputs + '\\' + 'Data_Types.xlsx'

file_HPI_repl_check_Sample = Dir_Results + '\\' +f'HPI_repl_check_sample.xlsx'

file_All_Data = Dir_Results+'\SFR_PD_Data_All_Monthly.pkl'

##########################################################################
################# Model File Functions ###################################
###########################################################################
def File_Splitted_Samples_func(Dir_Model):
    return {'OOT': Dir_Model + '\\'+'Sample_OOT.csv', 'OOS': Dir_Model + '\\'+'Sample_OOS.csv','TRN': Dir_Model + '\\'+'Sample_TRN.csv'}

def file_Num_WOE_GiniBinning_func(Dir_Model):
    return Dir_Model + '\\'+'WOE_Binning_Numeric_Drivers.csv'


def file_Insured_Num_WOE_GiniBinning_func(Dir_Model):
    return Dir_Model + '\\'+'WOE_Binning_Insured_Numeric_Drivers.csv'

def file_UnInsured_Num_WOE_GiniBinning_func(Dir_Model):
    return Dir_Model + '\\'+'WOE_Binning_UnInsured_Numeric_Drivers.csv'


def file_Cat_WOE_NoBinning_func(Dir_Model):
    return Dir_Model + '\\'+'WOE_NoBinning_Categoric_Drivers.csv'

def file_Cat_WOE_WoeDiffBinning_func(Dir_Model):
    return Dir_Model + '\\'+'WOE_Binning_Categoric_Drivers.csv'

def file_Num_WOE_GiniBinning_ADJ_func(Dir_Model):
    return Dir_Model + '\\'+'WOE_Binning_Numeric_Drivers_Adj.csv'

def file_Num_WOE_GiniBinning_Final_func(Dir_Model):
    return Dir_Model + '\\'+'WOE_Binning_Numeric_Drivers_Final.csv'

def file_Cat_WOE_WoeDiffBinning_ADJ_func(Dir_Model):
    return Dir_Model + '\\'+'WOE_Binning_Categoric_Drivers_Adj.csv'

def file_Cat_WOE_WoeDiffBinning_Final_func(Dir_Model):
    return Dir_Model + '\\'+'WOE_Binning_Categoric_Drivers_Final.csv'

def File_Splitted_WOE_func(Dir_Model): 
    return {'OOT': Dir_Model+'\\'+f'Sample_OOT_WOE.csv',
                                'OOS': Dir_Model+'\\'+f'Sample_OOS_WOE.csv',
                                'TRN': Dir_Model+'\\'+f'Sample_TRN_WOE.csv'} 

def File_Splitted_WOE_Final_func(Dir_Model): 
    return {'OOT': Dir_Model+'\\'+f'Sample_OOT_WOE_Final.csv',
                                'OOS': Dir_Model+'\\'+f'Sample_OOS_WOE_Final.csv',
                                'TRN': Dir_Model+'\\'+f'Sample_TRN_WOE_Final.csv'} 

def File_Pre_Sampling_WOE_func(Dir_Model):
    return Dir_Model+'\\'+f'Pre_Sampling_WOE.csv'

def file_univariate_analysis_func(Dir_Model):
    return Dir_Model+'\\'+ f'Univariate_Analysis.xlsx'

def file_Model_Multivariate_func(Dir_Model):
    return Dir_Model + '\\' + 'Res_MultivariateAnalysis.pkl'

def file_Res_MultiTesting_func(Dir_Model):
    return Dir_Model + '\\' + 'Res_MultiTesting.csv'

def file_LTV_Multi_func(Dir_Model):
    return Dir_Model + '\\' + 'LTV_Multi_Mdl.pkl'

def file_Scoring_Mdl_func(Dir_Model):
    return Dir_Model + '\\' + 'Final_Scoring_Mdl.pkl'

def seg_index_func(Covid_Inclusion_InSample, dlq_folder, Segment):
    return f'{Covid_Inclusion_InSample}'+'_'+f'{dlq_folder}'+'_' +f'{Segment}'

def file_MRS_Data_func(Dir_Model):
    return Dir_Model + '\\' + 'MRS_Data.pkl'

def file_MRS_Model_func(Dir_Model):
    return Dir_Model + '\\' + 'MRS_Model.pkl'

def file_MOC_func(Dir_Model):
    return Dir_Model + '\\MOC\\MOC.pkl' 

def file_param_model_func(Dir_Model):
    return Dir_Model + '\\Param_Model.pkl' 

dir_seg = {seg_index_func(Covid_Inclusion_InSample, dlq_folder, Segment): dir_for_model_func(Root_Dir_for_Model,Covid_Inclusion_InSample, dlq_folder, Segment) for Covid_Inclusion_InSample in Covid_Inclusion_InSample_full_list for dlq_folder in dlq_folder_full_list for Segment in  Segment_full_list}

file_Model_Multivariate_Seg = {seg_index_func(Covid_Inclusion_InSample, dlq_folder, Segment): file_Model_Multivariate_func(dir_for_model_func(Root_Dir_for_Model,Covid_Inclusion_InSample, dlq_folder, Segment)) for Covid_Inclusion_InSample in Covid_Inclusion_InSample_full_list for dlq_folder in dlq_folder_full_list for Segment in  Segment_full_list}

File_Splitted_WOE_Seg = {seg_index_func(Covid_Inclusion_InSample, dlq_folder, Segment): File_Splitted_WOE_func(dir_for_model_func(Root_Dir_for_Model,Covid_Inclusion_InSample, dlq_folder, Segment)) for Covid_Inclusion_InSample in Covid_Inclusion_InSample_full_list for dlq_folder in dlq_folder_full_list for Segment in Segment_full_list }


File_Splitted_WOE_Final_Seg = {seg_index_func(Covid_Inclusion_InSample, dlq_folder, Segment): File_Splitted_WOE_Final_func(dir_for_model_func(Root_Dir_for_Model,Covid_Inclusion_InSample, dlq_folder, Segment)) for Covid_Inclusion_InSample in Covid_Inclusion_InSample_full_list for dlq_folder in dlq_folder_full_list for Segment in Segment_full_list }

file_res_MultiTesting_Seg = {seg_index_func(Covid_Inclusion_InSample, dlq_folder, Segment): file_Res_MultiTesting_func(dir_for_model_func(Root_Dir_for_Model,Covid_Inclusion_InSample, dlq_folder, Segment)) for Covid_Inclusion_InSample in Covid_Inclusion_InSample_full_list for dlq_folder in dlq_folder_full_list for Segment in  Segment_full_list}

File_LTV_Multi_Seg = {seg_index_func(Covid_Inclusion_InSample, dlq_folder, Segment): file_LTV_Multi_func(dir_for_model_func(Root_Dir_for_Model,Covid_Inclusion_InSample, dlq_folder, Segment)) for Covid_Inclusion_InSample in Covid_Inclusion_InSample_full_list for dlq_folder in dlq_folder_full_list for Segment in  Segment_full_list}


File_Scoring_Mdl_Seg = {seg_index_func(Covid_Inclusion_InSample, dlq_folder, Segment): file_Scoring_Mdl_func(dir_for_model_func(Root_Dir_for_Model,Covid_Inclusion_InSample, dlq_folder, Segment)) for Covid_Inclusion_InSample in Covid_Inclusion_InSample_full_list for dlq_folder in dlq_folder_full_list for Segment in  Segment_full_list}


File_MRS_Data_Seg = {seg_index_func(Covid_Inclusion_InSample, dlq_folder, Segment): file_MRS_Data_func(dir_for_model_func(Root_Dir_for_Model,Covid_Inclusion_InSample, dlq_folder, Segment)) for Covid_Inclusion_InSample in Covid_Inclusion_InSample_full_list for dlq_folder in dlq_folder_full_list for Segment in  Segment_full_list}

File_MRS_Mdl_Seg = {seg_index_func(Covid_Inclusion_InSample, dlq_folder, Segment): file_MRS_Model_func(dir_for_model_func(Root_Dir_for_Model,Covid_Inclusion_InSample, dlq_folder, Segment)) for Covid_Inclusion_InSample in Covid_Inclusion_InSample_full_list for dlq_folder in dlq_folder_full_list for Segment in  Segment_full_list}
  


##########################################################################
################# Functions  on New Variables ###################################
###########################################################################

def add_new_var(df):
    #  Add new variables:  Combo_Province_Metro, term_age_tiers, Remaining_Term_over_Term
    print('***add combo province metro variable***')
    df['Combo_Province_Metro'] =  '' #default to be missing

    #df.loc[(df['Province'].isin(['BC'])& (~df['Metro_Region_BF_FMT'].isin(['Vancouver','Victoria'])))|df['Province'].isin(['YT']),'Combo_Province_Metro'] = 'BC_YT_Excl_Vancouver_Victoria'
    df.loc[(df['Province'].isin(['BC'])) & (~df['Metro_Region_BF_FMT'].isin(['Vancouver','Victoria'])),'Combo_Province_Metro'] = 'BC_Excl_Vancouver_Victoria'
    df.loc[df['Metro_Region_BF_FMT'].isin(['Vancouver']),'Combo_Province_Metro'] = 'Vancouver'
    df.loc[df['Metro_Region_BF_FMT'].isin(['Victoria']),'Combo_Province_Metro'] = 'Victoria'

    df.loc[(df['Province'].isin(['ON'])) & (~df['Metro_Region_BF_FMT'].isin(['Toronto'])),'Combo_Province_Metro'] = 'ON_except_Toronto'
    df.loc[df['Metro_Region_BF_FMT'].isin(['Toronto']),'Combo_Province_Metro'] = 'Toronto'
    
    df.loc[(df['Province'].isin(['QC'])) & (~df['Metro_Region_BF_FMT'].isin(['Montreal'])),'Combo_Province_Metro'] = 'QC_except_Montreal'
    df.loc[df['Metro_Region_BF_FMT'].isin(['Montreal']),'Combo_Province_Metro'] = 'Montreal'

    df.loc[(df['Province'].isin(['AB'])) & (~df['Metro_Region_BF_FMT'].isin(['Calgary','Edmonton'])),'Combo_Province_Metro'] = 'AB_Excl_Calgary_Edmonton'
    df.loc[df['Metro_Region_BF_FMT'].isin(['Calgary']),'Combo_Province_Metro'] = 'Calgary'
    df.loc[df['Metro_Region_BF_FMT'].isin(['Edmonton']),'Combo_Province_Metro'] = 'Edmonton'


    df.loc[df['Province'].isin(['YT','PE','NS','NB','NL','MB','SK','NT','NU']),'Combo_Province_Metro'] = df['Province']
    
    #df.loc[((df['Province'].isin(['ON'])) & (~df['Metro_Region_BF_FMT'].isin(['Toronto'])))| (df['Metro_Region_BF_FMT'].isin(['Vancouver','Victoria'])),'Combo_Province_Metro'] = 'ON_except_Toronto_or_Vancouver_Victoria'

    #df.loc[df['Province'].isin(['PE','NS','NB','NL']),'Combo_Province_Metro'] = 'Atlantic_PE_NS_NB_NL'

    #df.loc[df['Province'].isin(['AB']),'Combo_Province_Metro'] = 'AB'

    #df.loc[df['Province'].isin(['MB','SK','QC','NT','NU']),'Combo_Province_Metro'] = 'MB_SK_QC_NT_NU'
    #df.loc[df['Province'].isin(['QC']),'Combo_Province_Metro'] = 'QC'
    #df.loc[df['Province'].isin(['MB','SK','NT','NU']),'Combo_Province_Metro'] = 'MB_SK_NT_NU'

    # Add Combo_Term_Age
    # Not to be added because term is mortgage term in current contract, while age is since loan inception. Therefore, term can be greater than age. THe combo is not very meaningful in predicting risk.

    #  Add new variables:  Combo_Province_Metro_Override
    print('***add combo province metro variable with business override***')
    df['Combo_Province_Metro_Override'] =  '' #default to be missing

    #df.loc[df['Province'].isin(['AB','SK','MB','QC']),'Combo_Province_Metro_Override'] = 'Combo_Region_businessOverride_bin0'
    #df.loc[df['Province'].isin(['NB','NS','ON','NL','BC','YT','PE','NT']),'Combo_Province_Metro_Override'] = 'Combo_Region_businessOverride_bin1'

    # df.loc[~df['Province'].isin(['ON','BC']),'Combo_Province_Metro_Override'] = 'Combo_Region_businessOverride_bin0'
    # df.loc[df['Province'].isin(['ON','BC']),'Combo_Province_Metro_Override'] = 'Combo_Region_businessOverride_bin1'

    #finalized grouping after discussion on September 22, 2023.Rationale: small provinces / territories have lower macroeconomic stability (to this we would need some facts to support when documenting), and therefore more risky. QC is shown as low risk based on CBA, however internal data supports higher risk, and so it cannot go into the best bin
    df.loc[df['Province'].isin(['AB','SK','MB','YT','PE','NT','NU']),'Combo_Province_Metro_Override'] = 'AB_SK_MB_YT_PE_NT_NU'
    df.loc[df['Province'].isin(['QC','NB','NS','NL']),'Combo_Province_Metro_Override'] = 'QC_NB_NS_NL'
    # df.loc[df['Province'].isin(['ON','BC']),'Combo_Province_Metro_Override'] = 'ON_BC'
    df.loc[df['Province'].isin(['ON']),'Combo_Province_Metro_Override'] = 'ON'
    df.loc[df['Province'].isin(['BC']),'Combo_Province_Metro_Override'] = 'BC'

    # Add LoanPurpose_Override
    print('***add loan purpose override***')

    df['LoanPurpose_Override'] = df['LoanPurpose'].copy()

    df.loc[df['LoanPurpose'].isin(['',np.nan,'nan',str(np.nan),'OT']),'LoanPurpose_Override'] = np.nan

    df.loc[df['LoanPurpose'].isin(['TF','SR']),'LoanPurpose_Override'] = 'TF_or_SR'

    # Add Remaining_term/term as a way to normalize remaining term

    print('***add normalized remaining term***')

    df['Remaining_Term_over_Term'] = df['Remaining_Term']/df['Term_At_SL']

    return df


def add_combo_driver(df, for_var_selection = False, file_Insured_Num_WOE_GiniBinning = None, file_UnInsured_Num_WOE_GiniBinning = None):

       # Add Combo Driver
       # for_var_selection: Default False. Final combo driver (through Business override) was added for final scoring model and model application purpose (such as segmentation and impact analysis, or implementation). When this is true, combo drivers were added for univariate and multivariate analyses - and hence all based on automatic binning
       # file_Insured_Num_WOE_GiniBinning is the WOE file (csv) for binning based on insured data only
       # file_UnInsured_Num_WOE_GiniBinning is the WOE file (csv) for binning based on not insured data only


    if for_var_selection:

        # construct combo drivers based on automatic binning, only LTV including partner and HELOC (total)
        LTV_Tot_Var = df.columns[df.columns.str.contains('LTV_Tot')]

        woe_summary_insured = pd.DataFrame()
        woe_summary_uninsured = pd.DataFrame()

        if file_Insured_Num_WOE_GiniBinning is not None:
            woe_summary_insured = pd.read_csv(file_Insured_Num_WOE_GiniBinning, low_memory=False) 
            woe_summary_insured = woe_summary_insured.query('Bin!=-9999').copy()

        if file_UnInsured_Num_WOE_GiniBinning is not None:
            woe_summary_uninsured = pd.read_csv(file_UnInsured_Num_WOE_GiniBinning, low_memory=False) 
            woe_summary_uninsured = woe_summary_uninsured.query('Bin!=-9999').copy()


        for col_dr in LTV_Tot_Var:

            print(f'****Add Combo Insured Driver for {col_dr}: ****')

            col_combo_dr =  'Combo_Insured_'+col_dr
            df[col_combo_dr]  = ''  #default empty - will be applied to if either insured_ind or LTV missing

            woe_table_insured = woe_summary_insured.query('Variable_Name =="'+col_dr+'"')

            woe_table_uninsured = woe_summary_uninsured.query('Variable_Name =="'+col_dr+'"')

            if (not woe_table_insured.empty) and (not woe_table_uninsured.empty):
                # Insured
                interval_insured = sorted(list(set(list(woe_table_insured['Bin_Range_UB'].dropna()) + [-np.inf])))
                loc = (df['Insured_Ind']=='Insured')&(~(df[col_dr].isna()))   # for insured, and LTV available
                df.loc[loc,col_combo_dr] = df.loc[loc, col_dr].apply(lambda x: 'Insured; ' + str(round(max([y for y in interval_insured if y<x]),2)) + ' < ' + col_dr + ' <= ' + str(round(min([y for y in interval_insured if y>=x]),2)))

                # Not Insured
                interval_uninsured = sorted(list(set(list(woe_table_uninsured['Bin_Range_UB'].dropna()) + [-np.inf])))
                loc = (df['Insured_Ind']=='Not Insured')&(~(df[col_dr].isna()))   # for insured, and LTV available
                df.loc[loc,col_combo_dr] = df.loc[loc, col_dr].apply(lambda x: 'Uninsured; ' + str(round(max([y for y in interval_uninsured if y<x]),2)) + ' < ' + col_dr + ' <= ' + str(round(min([y for y in interval_uninsured if y>=x]),2)))


    else: 
        print('**************add combo LTV Insured Indicator driver')

        ############### agreed with business - 20231004 ###################
        df['Combo_LTV_Insured_Ind'] =  ''
        df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.40)& (~df['BF_LTV_Tot_Exp_FSA_Dw_WF'].isna())),'Combo_LTV_Insured_Ind'] = 'Insured, LTV<=0.40'
        df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.40)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.60)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.40<LTV<=0.60'
        df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.60)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.80)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.60<LTV<=0.80'
        df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.80)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.95)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.80<LTV<=0.95'
        #df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.60)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.95)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.60<LTV<=0.95'
        df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']> 0.95) & (~df['BF_LTV_Tot_Exp_FSA_Dw_WF'].isna())),'Combo_LTV_Insured_Ind'] = 'Insured, LTV>0.95'

        df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.30)& (~df['BF_LTV_Tot_Exp_FSA_Dw_WF'].isna())),'Combo_LTV_Insured_Ind'] = 'Uninsured, LTV<=0.30'
        df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.30)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.50)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.30<LTV<=0.50'
        df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.50)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.65)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.50<LTV<=0.65'
        df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.65)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.70)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.65<LTV<=0.70'
        df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.70)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.80)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.70<LTV<=0.80'
        df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']> 0.80)),'Combo_LTV_Insured_Ind'] = 'Uninsured, LTV>0.80'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.80)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.95)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.80<LTV<=0.95' #added to count for stress case where values decline dramatically
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']> 0.95) & (~df['BF_LTV_Tot_Exp_FSA_Dw_WF'].isna())),'Combo_LTV_Insured_Ind'] = 'Uninsured, LTV>0.95'

        
        ############### insured LTV 0.99 cutoff###################
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.392825)& (~df['BF_LTV_Tot_Exp_FSA_Dw_WF'].isna())),'Combo_LTV_Insured_Ind'] = 'Insured, LTV<=0.39'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.392825)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.575592)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.39<LTV<=0.58'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.575592)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.989426)),'Combo_LTV_Insured_Ind'] = 'Insured,0.58<LTV<=0.99'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']> 0.989426)),'Combo_LTV_Insured_Ind'] = 'Insured,LTV>0.99'

        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.30137)& (~df['BF_LTV_Tot_Exp_FSA_Dw_WF'].isna())),'Combo_LTV_Insured_Ind'] = 'Uninsured, LTV<=0.30'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.30137)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.503845)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.30<LTV<=0.50'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.503845)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.640042)),'Combo_LTV_Insured_Ind'] = 'Uninsured,0.50<LTV<=0.64'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.640042)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.800927)),'Combo_LTV_Insured_Ind'] = 'Uninsured,0.64<LTV<=0.80'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.800927)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.832027)),'Combo_LTV_Insured_Ind'] = 'Uninsured,0.80<LTV<=0.83'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']> 0.832027)),'Combo_LTV_Insured_Ind'] = 'Uninsured,LTV>0.83'

        ############### insured LTV 0.95 cutoff###################
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.40)& (~df['BF_LTV_Tot_Exp_FSA_Dw_WF'].isna())),'Combo_LTV_Insured_Ind'] = 'Insured, LTV<=0.40'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.40)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.60)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.40<LTV<=0.60'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.60)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.95)),'Combo_LTV_Insured_Ind'] = 'Insured,0.60<LTV<=0.95'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']> 0.95)),'Combo_LTV_Insured_Ind'] = 'Insured,LTV>0.95'

        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.30)& (~df['BF_LTV_Tot_Exp_FSA_Dw_WF'].isna())),'Combo_LTV_Insured_Ind'] = 'Uninsured, LTV<=0.30'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.30)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.50)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.30<LTV<=0.50'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.50)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.65)),'Combo_LTV_Insured_Ind'] = 'Uninsured,0.50<LTV<=0.65'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.65)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.80)),'Combo_LTV_Insured_Ind'] = 'Uninsured,0.65<LTV<=0.80'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.80)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.85)),'Combo_LTV_Insured_Ind'] = 'Uninsured,0.80<LTV<=0.85'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']> 0.85)),'Combo_LTV_Insured_Ind'] = 'Uninsured,LTV>0.85'

        ######### insured uninsured same LTV, most granular#############
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.30)& (~df['BF_LTV_Tot_Exp_FSA_Dw_WF'].isna())),'Combo_LTV_Insured_Ind'] = 'Insured, LTV<=0.30'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.30)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.40)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.30<LTV<=0.40'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.40)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.50)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.40<LTV<=0.50'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.50)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.60)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.50<LTV<=0.60'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.60)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.65)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.60<LTV<=0.65'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.65)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.70)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.65<LTV<=0.70'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.70)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.75)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.70<LTV<=0.75'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.75)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.80)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.75<LTV<=0.80'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.80)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.85)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.80<LTV<=0.85'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.85)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.95)),'Combo_LTV_Insured_Ind'] = 'Insured, 0.85<LTV<=0.95'
        # df.loc[((df['Insured_Ind']=='Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']> 0.95)),'Combo_LTV_Insured_Ind'] = 'Insured, LTV>0.95'

        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.30)& (~df['BF_LTV_Tot_Exp_FSA_Dw_WF'].isna())),'Combo_LTV_Insured_Ind'] = 'Uninsured, LTV<=0.30'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.30)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.40)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.30<LTV<=0.40'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.40)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.50)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.40<LTV<=0.50'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.50)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<= 0.60)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.50<LTV<=0.60'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.60)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.65)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.60<LTV<=0.65'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.65)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.70)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.65<LTV<=0.70'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.70)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.75)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.70<LTV<=0.75'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.75)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.80)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.75<LTV<=0.80'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.80)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.85)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.80<LTV<=0.85'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF'] > 0.85)& (df['BF_LTV_Tot_Exp_FSA_Dw_WF']<=  0.95)),'Combo_LTV_Insured_Ind'] = 'Uninsured, 0.85<LTV<=0.95'
        # df.loc[((df['Insured_Ind']=='Not Insured') & (df['BF_LTV_Tot_Exp_FSA_Dw_WF']> 0.95)),'Combo_LTV_Insured_Ind'] = 'Uninsured, LTV>0.95'

        # add Beacon BNI combo driver to test the model performance. Added: Oct 17 2023
        # print('***add Beacon BNI combo driver***')

        # df['Combo_Beacon_BNI'] =  ''

        # Beacon_cutoff = [299,550,600,640,680,720,760,820,900]
        
        # for b1 in range(1,len(Beacon_cutoff)):
        #     if Beacon_cutoff[b1] == 550:
        #         BNI_cutoff = [0,650, 920, 999]

        #     if Beacon_cutoff[b1] == 600:
        #         BNI_cutoff = [0,999]

        #     if Beacon_cutoff[b1] == 640:
        #         #BNI_cutoff = [0,650, 920, 999]
        #         BNI_cutoff = [0,650, 999]

        #     if Beacon_cutoff[b1] == 680:
        #         #BNI_cutoff = [0,650, 920, 999]
        #         BNI_cutoff = [0,650, 999]

        #     if Beacon_cutoff[b1] == 720:
        #         #BNI_cutoff = [0,650, 920, 999]
        #         BNI_cutoff = [0,920, 999]

        #     if Beacon_cutoff[b1] == 760:
        #         BNI_cutoff = [0,920, 999]

        #     if Beacon_cutoff[b1] == 820:
        #         #BNI_cutoff = [0, 970, 999]
        #         BNI_cutoff = [0, 999]

        #     if Beacon_cutoff[b1] == 900:
        #         #BNI_cutoff = [0,980, 999]
        #         BNI_cutoff = [0, 999]

        #     for b2 in range(1,len(BNI_cutoff)):
        #         loc = (df['Beacon_Avg_App_CoApp']>Beacon_cutoff[b1-1]) & (df['Beacon_Avg_App_CoApp']<=Beacon_cutoff[b1]) & (df['BNI_Avg_App_CoApp']>BNI_cutoff[b2-1]) & (df['BNI_Avg_App_CoApp']<=BNI_cutoff[b2])
        #         cat_name = str(Beacon_cutoff[b1-1]) + '<Beacon<=' + str(Beacon_cutoff[b1]) + ',' + str(BNI_cutoff[b2-1]) + '<BNI<=' + str(BNI_cutoff[b2])
        #         df.loc[loc,'Combo_Beacon_BNI'] = cat_name
        
    return df


#%%################################################################################################################
#                                    General Functions
##################################################################################################################



def del_all_var_Same_obj(t):
    # delete all variables that share the same object with variable t

    l = [k for k,v in globals().items() if v is t]  #list all variabels sharing the same object

    for i in l:
        del(globals()[i])   #delete those variables

    return None


def driver_distribution_over_time(df = None, var = None, fig_Dir = None, fig_name_suffix = None):
    # Plot driver distribution over time (stacked area plot)

    print('***Plot '+f'{var}'+'***')
    df_plot = df.groupby(col_snap_date)[var].value_counts(normalize = True,dropna=False, sort = True).unstack()
    df_plot.plot(kind='area',stacked=True).legend(bbox_to_anchor = (1,1))
    plt.xlabel('Date')
    plt.ylabel(var + ' Bin Relative Frequency')
    plt.title(var + ' Distribution ' + fig_name_suffix)
    # plt.show()
    plt.legend(loc='center',)
    # plt.savefig(fig_Dir+'\\'+var+fig_name_suffix+'.png')
    plt.show()


def driver_woe_distribution(df_input = None, df_WOE_cat = None, df_WOE_num = None, all_col = None, Dir_out = None, var_group = None):
    # Plot driver (woe) distribution changes over time

    def driver_WOE_distribution_wo_group(df_analysis = None, fig_name_suffix = '', cols_all = None, cols_num = None, cols_cat = None, df_WOE_cat = None, df_WOE_num = None):

        df_WOE = df_analysis[[col_snap_date] + [var+'_WOE' for var in cols_all]].copy().round(6)

        df_WOE_cat = df_WOE_cat[df_WOE_cat['Variable_Name'].isin(cols_cat)].copy().round(6)
        df_WOE_num = df_WOE_num[df_WOE_num['Variable_Name'].isin(cols_num)].copy().round(6)

        for var in cols_num:
            df = df_WOE_num[df_WOE_num['Variable_Name']==var][['Bin_WOE','Bin_Range']].copy()
            df = df.set_index('Bin_WOE')
            mapping_dict = df.to_dict(orient='dict')['Bin_Range'].copy()
            df_WOE.loc[:,var] = df_WOE[var+'_WOE'].copy().map(mapping_dict)

        for var in cols_cat:
            df = df_WOE_cat[df_WOE_cat['Variable_Name']==var][['WOE','Bin']].copy()
            df = df.set_index('WOE')
            mapping_dict = df.to_dict(orient='dict')['Bin'].copy()

            df_WOE.loc[:,var] = df_WOE[var+'_WOE'].copy().map(mapping_dict)

        for var in cols_all:
            driver_distribution_over_time(df = df_WOE, var = var, fig_Dir = Dir_out, fig_name_suffix = fig_name_suffix)

    cols_skip_for_driver_distribution  = cols_skip if var_group is None else  list(set(cols_skip+[var_group, var_group+'_WOE'])) 
    cols_all, cols_num, cols_cat = num_cat_col_split(df_input[all_col], cols_skip= cols_skip_for_driver_distribution)   

    if var_group is not None:
        group_list = df_input[var_group].dropna().unique()

        for seg in group_list:
            df_analysis = df_input[df_input[var_group]==seg]
            driver_WOE_distribution_wo_group(df_analysis = df_analysis, fig_name_suffix = ' for ' +var_group + '='+seg, cols_all=cols_all, cols_num = cols_num, cols_cat = cols_cat, df_WOE_cat = df_WOE_cat, df_WOE_num = df_WOE_num)

    else:
        driver_WOE_distribution_wo_group(df_analysis = df_input, fig_name_suffix = '',  cols_all=cols_all, cols_num = cols_num, cols_cat = cols_cat, df_WOE_cat = df_WOE_cat, df_WOE_num = df_WOE_num)


#%%################################################################################################################
#                                     Functions on Multivariate Analysis
##################################################################################################################

def get_low_correlated_cols_subsets(df, cols, high_corr_theshold=0.5):
    # functionality: 
    #   get the lists of columns among which the correlation is lower than the set threshold.
    # inputs: 
    #   df: the input dataframe
    #   cols: the candidate columns that are taken into account
    #   high_corr_theshold: the correlation threshold
    # outputs:
    #   res: the result dictionary
    def refine_clusters(cols_cluster_lsOfSet):
        for i in range(len(cols_cluster_lsOfSet)):
            for j in range(i+1, len(cols_cluster_lsOfSet)):
                if cols_cluster_lsOfSet[i].intersection(cols_cluster_lsOfSet[j]):
                    cols_cluster_lsOfSet[i] = cols_cluster_lsOfSet[i].union(cols_cluster_lsOfSet[j])
                    cols_cluster_lsOfSet = cols_cluster_lsOfSet[:j]+cols_cluster_lsOfSet[j+1:]
                    cols_cluster_lsOfSet = refine_clusters(cols_cluster_lsOfSet)
                    return cols_cluster_lsOfSet
        return cols_cluster_lsOfSet
    
    def find_dup_cols(corr):
        cols_dup= [{corr.columns[i], corr.columns[j]} for i in range(len(corr.index)) for j in range(i+1, len(corr.columns)) if corr.iloc[i, j]==1]
        cols_dup = refine_clusters(cols_dup)
        return cols_dup
    
    def cluster_cols(corr, cols_dup, high_corr_theshold):
        # he cluster is formed such that for each initial cluster, if any variable in the remaining has high correlation (>threshold) with any variable in the initial cluster, then this variable will be picked up and added to the initial cluster. In this way, an initial cluster grows bigger to the extent that no variable outside the cluster has high correlation with any of variables in the cluster. Similarly, other clusters are formed in a similar fashion.

        def corr_drop_dup(corr, cols_dup):
            for dups in cols_dup:
                cols_dup = list(dups)[1:]
                corr = corr.drop(index=cols_dup, columns=cols_dup)
            return corr
        def ini_cluster_cols(cols_grp):
            cols_cluster = []
            for key, val in cols_grp.items():
                val.add(key)
                cols_cluster.append(val)
            return cols_cluster
        
        corr = corr_drop_dup(corr, cols_dup)
        cols_grp = {col: set(corr[(abs(corr[col])>=high_corr_theshold)].index) for col in corr.columns}
        cols_cluster = ini_cluster_cols(cols_grp)
        cols_cluster = refine_clusters(cols_cluster)
        return cols_cluster
    
    def bt_uncorr_cols_sets(cols_correlated, idx, cols_select, res_bt):
        if idx==len(cols_correlated):
            res_bt.append(cols_select.copy())
            return res_bt
        for col in cols_correlated[idx]:
            bt_uncorr_cols_sets(cols_correlated, idx+1, cols_select+[col], res_bt)
        return res_bt
                    
    corr = univariate_numeric_corr_analysis(df, cols)['Corr_Matrix']['spearman']
    cols_dup = find_dup_cols(corr)
    cols_cluster = cluster_cols(corr, cols_dup, high_corr_theshold)
    lowCorr_ini_select = [list(x)[0] for x in cols_cluster if len(x)==1]
    cols_correlated = [list(x) for x in cols_cluster if len(x)>1] 
    print(f"Start Backtracking! ({len(cols_cluster)-len(lowCorr_ini_select)} candidates) ")
    start_time = time.time()
    res_bt = bt_uncorr_cols_sets(cols_correlated, 0, lowCorr_ini_select, [])
    print(f"Backtracking Done! ({datetime.timedelta(seconds=time.time()-start_time)})" )
    res = {'Lists': [list(x) for x in res_bt], 'Correlation': corr, 'Clusters': cols_cluster, 'Duplicates': cols_dup}
    return res


def univariate_gini(df_uni_gini=[], col_dr=[], col_target=[]):
    # functionality: 
    #   calculate the univariate gini.
    # inputs: 
    #   df_uni_gini: the input dataframe
    #   col_dr: the driver column
    #   col_target: the target column
    # outputs:
    #   gini: the univariate gini

    df_uni_gini = df_uni_gini.dropna(subset=[col_dr, col_target]).copy()

    auc = roc_auc_score(df_uni_gini[col_target], df_uni_gini[col_dr])
    gini = abs(2*auc-1)
    return gini

def cal_relative_weights(sm_fitted_model):
    # functionality: 
    #   calculate the relative weights for the drivers in the fitted model.
    # inputs: 
    #   sm_fitted_model: the fitted model
    # outputs:
    #   ss_rw_var: the series of the drivers' relative weights  
    pred_prob = sm_fitted_model.predict()
    pred_logodds = np.log(pred_prob/(1-pred_prob))
    std_logodds = pred_logodds.std()
    std_var = sm_fitted_model.model.exog.std(axis=0)
    coefv_on_std_var = abs(sm_fitted_model.params)* std_var/std_logodds;
    ss_rw_var = coefv_on_std_var/sum(coefv_on_std_var)  
    return ss_rw_var
# %%
def get_model_stats(sm_fitted_model, auc):
    # functionality: 
    #   get the model's statisitcal status.
    # inputs: 
    #   sm_fitted_model: the fitted model
    #   auc: the model's auc on the training data
    # outputs:
    #   res: the result dataframe  
    cols = sm_fitted_model.model.exog_names[1:]
    df_x = pd.DataFrame(sm_fitted_model.model.exog[:, 1:], columns=cols)
    df_y = pd.DataFrame(sm_fitted_model.model.endog, columns=['Target'])
    df_data = pd.concat([df_x, df_y], axis=1)
    res = pd.DataFrame()
    res['Feature'] = ['Intercept']+cols
    res['Standard Error'] = sm_fitted_model.bse.values
    res['tStat'] = sm_fitted_model.tvalues.values
    res['Coefficients'] = sm_fitted_model.params.values
    res['Relative Weight'] = cal_relative_weights(sm_fitted_model).values
    uni_gini = [np.nan]
    for col in cols:
        uni_gini.append(univariate_gini(df_data, col, 'Target'))
    res['Univariate_Gini'] = uni_gini
    res['p-value'] = sm_fitted_model.pvalues.values
    res = res[res['Coefficients']!=0]
    cols_select = [cols[i] for i in range(len(cols)) if sm_fitted_model.params[i+1]!=0]
    if len(cols_select)>1:
        res["VIF"] = [np.nan]+[variance_inflation_factor(df_x[cols_select].values, i) for i in range(len(cols_select))]
    else:
        res["VIF"] = [np.nan for _ in range(len(cols_select)+1)]
    res["Gini"] = 2*auc-1
    corr = univariate_numeric_corr_analysis(df_x, cols_select)['Corr_Matrix']['spearman']
    res = res.merge(corr, how='left', left_on='Feature', right_on=corr.index)
    res = res.sort_values('Relative Weight', ascending=False)
    res = res.reset_index(drop=True)                
    return res
# %%
def get_model_predicted_deciles(sm_fitted_model):  
    # functionality: 
    #   get the pd predictions deciles.
    # inputs: 
    #   sm_fitted_model: the fitted model
    # outputs:
    #   res: the result dataframe  
    df_res = pd.DataFrame([[x, y] for x, y in zip(sm_fitted_model.predict(), sm_fitted_model.model.endog)], columns=['Pred', 'Label'])
    df_res = df_res.sort_values('Pred')
    pred_decile = [0]+[df_res['Pred'].quantile(i/10) for i in range(1,11)]
    res = []
    for i in range(1,11):
        df_decile = df_res[(df_res['Pred']>pred_decile[i-1]) & (df_res['Pred']<=pred_decile[i])]
        res.append([i, df_decile['Pred'].sum(), df_decile['Label'].sum(), df_decile['Label'].count()])
    res = pd.DataFrame(res, columns=['Decile', 'Predicted', 'Observed', 'Count'])
    return res
# %%
def org_top_n_model_res(ls_auc_model, n=100000):
    # functionality: 
    #   organize the top n performing models
    # inputs: 
    #   auc_model: the list of [model auc, model]
    #   n: indicate the number of top performing model selected
    # outputs:
    #   res: the result dataframe  
    func = lambda x: x[0] if type(x[0]) in (float, int) else 0                
    ls_auc_model = sorted(ls_auc_model, key=func, reverse=True)[:n]
    model_unique = {}
    for auc_model in ls_auc_model:
        features = sorted(auc_model[1].model.exog_names[1:])
        key = ', '.join([str(len(features))]+features)
        if key not in model_unique:
            model_unique[key] = auc_model
    return model_unique
# %%
def feafure_selection_wLasso(df, col_target, list_cols, n_feature_select, n_lambda=10, min_lambda=1e-4, max_lambda=1e4):
    # functionality: 
    #   conduct feature selection using lasso
    # inputs: 
    #   df: the training dataframe
    #   col_target: the target column
    #   list_cols: the list of candidate columns
    #   n_feature_select: the max number of selected features
    #   n_lambda: the number of candidate lambdas
    #   min_lambda: the minimum lambda
    #   max_lambda: the maximum lambda    
    # outputs:
    #   res: the selected models using lasso
    print('feafure_selection_wLasso')
    def get_best_lasso_model(df, col_target, list_cols, n_feature_select, n_lambda, min_lambda, max_lambda):
        ss_y = df[col_target]
        lst_lambda = [10**(math.log10(min_lambda)+i*(math.log10(max_lambda)-math.log10(min_lambda))/(n_lambda-1)) for i in range(n_lambda)]
        auc_model = []
        for i, cols_cand in enumerate(list_cols):
            print(f'feafure_selection_wLasso -> {i}/{len(list_cols)} ({n_feature_select})')
            df_x  = df[cols_cand]
            for c in lst_lambda:
                mdl_lr_l1 = sm.Logit(ss_y, sm.add_constant(df_x, has_constant='add')).fit_regularized(disp=0, alpha=c, trim_mode='size', qc_tol=1)
                cols_select_l1 = [cols_cand[i] for i in range(len(cols_cand)) if mdl_lr_l1.params[i+1]!=0]
                if len(cols_select_l1)<=n_feature_select:
                    mdl_lr_l1 = sm.Logit(ss_y, sm.add_constant(df_x[cols_select_l1], has_constant='add')).fit(disp=0)
                    auc = roc_auc_score(ss_y, mdl_lr_l1.predict())
                    auc_model.append([auc, mdl_lr_l1])
                    break
        return auc_model
    auc_model = get_best_lasso_model(df, col_target, list_cols, n_feature_select, n_lambda, min_lambda, max_lambda)
    res = org_top_n_model_res(auc_model)
    return res
# %%
def feafure_selection_stepwise_parallel_Alt(df, col_target, list_cols, direction='forward', n_feature_select='auto'):
    # functionality: 
    #   conduct feature selection using stepwise approach
    # inputs: 
    #   df: the training dataframe
    #   col_target: the target column
    #   list_cols: the list of candidate columns
    #   direction: the dirction of the stepwise approach
    #   n_feature_select: the max number of selected features
    #   n_shuffle: the number of shuffles of the candidate columns
    # outputs:
    #   res: the selected models using stepwise approach
    print('feafure_selection_stepwise')

    def get_best_stepwise_model(col_target, cols_cand, n_feature_select, direction):
        ss_y = df[col_target]
        df_x  = df[cols_cand]
        if n_feature_select=='auto' or n_feature_select<len(cols_cand):
            mdl_lr = LogisticRegression(penalty='none', max_iter=int(1e5))
            mdl_stepwise = SequentialFeatureSelector(mdl_lr, n_features_to_select=n_feature_select, 
            cv=5, scoring='roc_auc', direction=direction,n_jobs=-1).fit(df_x, ss_y)
            cols_select_stepwise = df_x.columns[mdl_stepwise.support_].to_list()
        else:
            cols_select_stepwise = cols_cand
        mdl_stepwise = sm.Logit(ss_y, sm.add_constant(df_x[cols_select_stepwise], has_constant='add')).fit(disp=0)
        auc = roc_auc_score(ss_y, mdl_stepwise.predict())
        return [auc, mdl_stepwise]

    auc_model = Parallel(n_jobs=-1)(delayed(get_best_stepwise_model)(col_target, cols_cand,n_feature_select, direction) for cols_cand in tqdm(list_cols))

    res = org_top_n_model_res(auc_model)
    return res

#%%

def feafure_selection_stepwise(df, col_target, list_cols, direction='forward', n_feature_select='auto'):
    # functionality: 
    #   conduct feature selection using stepwise approach
    # inputs: 
    #   df: the training dataframe
    #   col_target: the target column
    #   list_cols: the list of candidate columns
    #   direction: the dirction of the stepwise approach
    #   n_feature_select: the max number of selected features
    #   n_shuffle: the number of shuffles of the candidate columns
    # outputs:
    #   res: the selected models using stepwise approach
    print('feafure_selection_stepwise')
    def get_best_stepwise_model(df, col_target, list_cols, n_feature_select, direction):
        ss_y = df[col_target]
        auc_model_1 = []
        for i, cols_cand in enumerate(list_cols):
            # logger = logging.getLogger()
            # logger.info(f'feafure_selection_stepwise -> {i}/{len(list_cols)} ({direction} {n_feature_select})')
            print(f'feafure_selection_stepwise -> {i}/{len(list_cols)} ({direction} {n_feature_select})')

            df_x  = df[cols_cand]
            if n_feature_select=='auto' or n_feature_select<len(cols_cand):
                mdl_lr = LogisticRegression(penalty='none', max_iter=int(1e5))
                mdl_stepwise = SequentialFeatureSelector(mdl_lr, n_features_to_select=n_feature_select, 
                                                         cv=5, scoring='roc_auc', direction=direction,n_jobs=-1).fit(df_x, ss_y)
                cols_select_stepwise = df_x.columns[mdl_stepwise.support_].to_list()
            else:
                cols_select_stepwise = cols_cand
            mdl_stepwise = sm.Logit(ss_y, sm.add_constant(df_x[cols_select_stepwise], has_constant='add')).fit(disp=0)
            auc = roc_auc_score(ss_y, mdl_stepwise.predict())
            auc_model_1.append([auc, mdl_stepwise])
        return auc_model_1

    n_cols = len(list_cols)

    index_list = np.linspace(0,n_cols, num=num_cores +1,dtype=int, endpoint=True)

    list_cols_sep = []

    for i_index_list in range(0,len(index_list)-1):
        list_cols_sep.append(list_cols[index_list[i_index_list]:index_list[i_index_list+1]])



    auc_model_list = Parallel(n_jobs=num_cores)(delayed(get_best_stepwise_model)(df, col_target, list_cols_1, n_feature_select, direction) for list_cols_1 in list_cols_sep )
    # auc_model_list = [get_best_stepwise_model(df, col_target, list_cols_1, n_feature_select, direction) for list_cols_1 in list_cols_sep]

    auc_model = []
    for i_auc_model_list in range(len(auc_model_list)):
        auc_model = auc_model + auc_model_list[i_auc_model_list]
    
    res = org_top_n_model_res(auc_model)
    return res    



#%%
def feafure_selection_stepwise_noparallel(df, col_target, list_cols, direction='forward', n_feature_select='auto'):
    # functionality: 
    #   conduct feature selection using stepwise approach
    # inputs: 
    #   df: the training dataframe
    #   col_target: the target column
    #   list_cols: the list of candidate columns
    #   direction: the dirction of the stepwise approach
    #   n_feature_select: the max number of selected features
    #   n_shuffle: the number of shuffles of the candidate columns
    # outputs:
    #   res: the selected models using stepwise approach
    print('feafure_selection_stepwise')
    def get_best_stepwise_model(df, col_target, list_cols, n_feature_select, direction):
        ss_y = df[col_target]
        auc_model = []
        for i, cols_cand in enumerate(list_cols):
            print(f'feafure_selection_stepwise -> {i}/{len(list_cols)} ({direction} {n_feature_select})')
            # if (i > 185 and n_feature_select==5) or (i>187 and n_feature_select==6) or (i>189 and n_feature_select ==7) or (i>190 and n_feature_select==8):
            df_x  = df[cols_cand]
            if n_feature_select=='auto' or n_feature_select<len(cols_cand):
                mdl_lr = LogisticRegression(penalty='none', max_iter=int(1e5))
                mdl_stepwise = SequentialFeatureSelector(mdl_lr, n_features_to_select=n_feature_select, 
                                                        cv=5, scoring='roc_auc', direction=direction,n_jobs=-1).fit(df_x, ss_y)
                cols_select_stepwise = df_x.columns[mdl_stepwise.support_].to_list()
            else:
                cols_select_stepwise = cols_cand
            mdl_stepwise = sm.Logit(ss_y, sm.add_constant(df_x[cols_select_stepwise], has_constant='add')).fit(disp=0)
            auc = roc_auc_score(ss_y, mdl_stepwise.predict())
            auc_model.append([auc, mdl_stepwise])
        return auc_model
    auc_model = get_best_stepwise_model(df, col_target, list_cols, n_feature_select, direction)
    res = org_top_n_model_res(auc_model)
    return res    

#%%
def feafure_selection_stepwise_noparallel_debug(df, col_target, list_cols, direction='forward', n_feature_select='auto'):
    # functionality: 
    #   conduct feature selection using stepwise approach
    # inputs: 
    #   df: the training dataframe
    #   col_target: the target column
    #   list_cols: the list of candidate columns
    #   direction: the dirction of the stepwise approach
    #   n_feature_select: the max number of selected features
    #   n_shuffle: the number of shuffles of the candidate columns
    # outputs:
    #   res: the selected models using stepwise approach
    print('feafure_selection_stepwise')
    def get_best_stepwise_model(df, col_target, list_cols, n_feature_select, direction):
        ss_y = df[col_target]
        auc_model = []
        for i, cols_cand in enumerate(list_cols):
            print(f'feafure_selection_stepwise -> {i}/{len(list_cols)} ({direction} {n_feature_select})')
            # if (i > 185 and n_feature_select==5) or (i>187 and n_feature_select==6) or (i>189 and n_feature_select ==7) or (i>190 and n_feature_select==8):
            df_x  = df[cols_cand]
            if n_feature_select=='auto' or n_feature_select<len(cols_cand):
                mdl_lr = LogisticRegression(penalty='none', max_iter=int(1e5))
                mdl_stepwise = SequentialFeatureSelector(mdl_lr, n_features_to_select=n_feature_select, 
                                                        cv=5, scoring='roc_auc', direction=direction,n_jobs=-1).fit(df_x, ss_y)
                cols_select_stepwise = df_x.columns[mdl_stepwise.support_].to_list()
            else:
                cols_select_stepwise = cols_cand
            mdl_stepwise = sm.Logit(ss_y, sm.add_constant(df_x[cols_select_stepwise], has_constant='add')).fit(disp=0)
            auc = roc_auc_score(ss_y, mdl_stepwise.predict())
            auc_model.append([auc, mdl_stepwise])
        return auc_model
    auc_model = get_best_stepwise_model(df, col_target, list_cols, n_feature_select, direction)
    res = org_top_n_model_res(auc_model)
    return res    



# %%
def feafure_selection_rfe(df, col_target, list_cols, n_feature_select):
    # functionality: 
    #   conduct feature selection using recursive feature elimination approach
    # inputs: 
    #   df: the training dataframe
    #   col_target: the target column
    #   list_cols: the list of candidate columns
    #   n_feature_select: the max number of selected features
    # outputs:
    #   res: the selected models using rfe
    print('feafure_selection_rfe')
    def get_best_rfe_model(df, col_target, list_cols, n_feature_select):
        ss_y = df[col_target]
        auc_model = []
        for i, cols_cand in enumerate(list_cols):
            print(f'feafure_selection_rfe -> {i}/{len(list_cols)} ({n_feature_select})')
            df_x  = df[cols_cand]
            if n_feature_select=='auto' or n_feature_select<len(cols_cand):
                mdl_lr = LogisticRegression(penalty='none', max_iter=int(1e5))
                mdl_rfe = RFE(mdl_lr, n_features_to_select=n_feature_select).fit(df_x, ss_y)
                cols_select_rfe = df_x.columns[mdl_rfe.support_].to_list()
            else:
                cols_select_rfe = cols_cand
            mdl_rfe = sm.Logit(ss_y, sm.add_constant(df_x[cols_select_rfe], has_constant='add')).fit(disp=0)
            auc = roc_auc_score(ss_y, mdl_rfe.predict())
            auc_model.append([auc, mdl_rfe])
        return auc_model
    auc_model = get_best_rfe_model(df, col_target, list_cols, n_feature_select)
    res = org_top_n_model_res(auc_model)
    return res
# %%
def multivariate_analysis(df, cols_lowCorr_lists, col_target, min_n_features=3, max_n_features=10, file_res='',dir_out=[]):
    # functionality: 
    #   conduct multivariate analysis/feature selection using lasso, stepwise approach and rfe
    # inputs: 
    #   df: the training dataframe
    #   cols_lowCorr_lists: the lists of columns with low correlations
    #   col_target: the target column
    #   min_n_features: the minimum number of selected features
    #   max_n_features: the maximum number of selected features
    #   list_cols: the list of candidate columns
    #   file_res: the pickle file name to save the multivariate analysis results
    # outputs:
    #   res: the selected models using each feature selection approach
    max_n_features = min(max_n_features, len(min(cols_lowCorr_lists, key=len)))
    if min_n_features>max_n_features:
        print(f"The number of drivers with low correlation is less than {min_n_features}!!!")
        return cols_lowCorr_lists
    res = {}
    for n_features in range(max_n_features, min_n_features-1, -1):
        # res_lasso = feafure_selection_wLasso(df, col_target, cols_lowCorr_lists, n_feature_select=n_features)
        # res_stepwise_fwd = feafure_selection_stepwise(df, col_target, cols_lowCorr_lists, direction='forward', n_feature_select=n_features)
        res_stepwise_bwd = feafure_selection_stepwise(df, col_target, cols_lowCorr_lists, direction='backward', n_feature_select=n_features)    
        # res_rfe = feafure_selection_rfe(df, col_target, cols_lowCorr_lists, n_feature_select=n_features)
        # for res_tmp in [res_lasso, res_stepwise_fwd, res_stepwise_bwd, res_rfe]:
        for res_tmp in [res_stepwise_bwd]:
            res.update(res_tmp)
    
    del res_stepwise_bwd   #release memory. otherwise may memory error

    df_mdl_stat = pd.DataFrame()
    i_mdl = 0
    for key, val in res.items():
        i_mdl = i_mdl + 1
        auc, mdl = val
        stat_model = get_model_stats(mdl, auc)
        res_decile = get_model_predicted_deciles(mdl)
        res_features = [x for x in stat_model["Feature"] if x!='Intercept']
        res[key] = {'Stat': stat_model, 'Deciles': res_decile, 'Model': mdl,  "Features": res_features, "Model_Number": i_mdl}
        res[key]['Stat']['Feature_Count'] = key.split(',',1)[0]
        res[key]['Stat']['Model_Nbr'] = i_mdl
        res[key]['Stat']['Model_Var'] = key.split(',',1)[1].strip()
        df_mdl_stat = pd.concat([df_mdl_stat, res[key]['Stat']])
        df_mdl_stat =  df_mdl_stat.reindex(columns = ['Model_Nbr','Feature_Count','Model_Var'] + list(df_mdl_stat.columns[~df_mdl_stat.columns.isin(['Model_Nbr','Feature_Count','Model_Var'])]))

 
    if len(dir_out)>0:
        df_mdl_stat.to_excel(dir_out+'\\'+'Multi_Var_Summary.xlsx',sheet_name ='Stat', index = False)

    del df_mdl_stat  # release memory space

    if len(file_res)>0:
        with open(file_res, 'wb') as file:
            pickle.dump(res, file)

    return res


#%%
def multivariate_analysis_parallel(df, cols_lowCorr_lists, col_target, min_n_features=4, max_n_features=10, file_res='',dir_out=[]):
    # This function has no issue running for some cases, but sometimes throw error "incorrect parameters", which I did not figure out why because
    # the error does not show up in many test cases and only appears in some cases. And, this error is not specific and could not find a helpful tip by searching internet. The internet provides some explanation for some other cases, not linked to my case
    # functionality: 
    #   conduct multivariate analysis/feature selection using lasso, stepwise approach and rfe
    # inputs: 
    #   df: the training dataframe
    #   cols_lowCorr_lists: the lists of columns with low correlations
    #   col_target: the target column
    #   min_n_features: the minimum number of selected features
    #   max_n_features: the maximum number of selected features
    #   list_cols: the list of candidate columns
    #   file_res: the pickle file name to save the multivariate analysis results
    # outputs:
    #   res: the selected models using each feature selection approach
    max_n_features = min(max_n_features, len(min(cols_lowCorr_lists, key=len)))
    if min_n_features>max_n_features:
        print(f"The number of drivers with low correlation is less than {min_n_features}!!!")
        return cols_lowCorr_lists
    res_tmp_list = Parallel(n_jobs=-1)([delayed(feafure_selection_stepwise)(df, col_target, cols_lowCorr_lists, direction='backward', n_feature_select=n_features) for n_features in range(max_n_features, min_n_features-1, -1)])
    res = {}
    for res_tmp in res_tmp_list:
        res.update(res_tmp)

    df_mdl_stat = pd.DataFrame()
    i_mdl = 0
    for key, val in res.items():
        i_mdl = i_mdl + 1
        auc, mdl = val
        stat_model = get_model_stats(mdl, auc)
        res_decile = get_model_predicted_deciles(mdl)
        res_features = [x for x in stat_model["Feature"] if x!='Intercept']
        res[key] = {'Stat': stat_model, 'Deciles': res_decile, 'Model': mdl,  "Features": res_features, "Model_Number": i_mdl}
        res[key]['Stat']['Feature_Count'] = key[0]
        res[key]['Stat']['Model_Nbr'] = i_mdl
        res[key]['Stat']['Model_Var'] = key[2:].strip()
        df_mdl_stat = pd.concat([df_mdl_stat, res[key]['Stat']])
        df_mdl_stat =  df_mdl_stat.reindex(columns = ['Model_Nbr','Model_Var'] + list(df_mdl_stat.columns[~df_mdl_stat.columns.isin(['Model_Nbr','Model_Var'])]))

    if len(file_res)>0:
        with open(file_res, 'wb') as file:
            pickle.dump(res, file)
    if len(dir_out)>0:
        df_mdl_stat.to_excel(dir_out+'\\'+'Multi_Var_Summary.xlsx',sheet_name ='Stat', index = False)
    return res


#%%################################################################################################################
#                                     Functions on Converting Data Types
##################################################################################################################

def convert_data_types(file_in_data_types,df):
# convert data types based on input file specification
    data_types = pd.read_excel(file_in_data_types)

    data_type_map = data_types.set_index('Variable').to_dict()['Type'].copy()

    data_type_map_short = {key: data_type_map[key] for key in list(data_type_map.keys()) if key in list(df.columns)}

    df_after_type_conversion = df.astype(data_type_map_short).copy()

    # for key,value in data_type_map_short.items():

    #     df = df.astype({key:value})

    return df_after_type_conversion

#%%################################################################################################################
#                                     Multiviarate Analysis Tests
##################################################################################################################


# %%
def add_model_prediction(df, model_sm_fitted):
    print('segment_model_testing -> add_model_prediction')
    col_target = model_sm_fitted.model.endog_names
    cols_dr = [x for x in model_sm_fitted.model.exog_names if x!='const']
    df["Pred_"+col_target] = model_sm_fitted.predict(sm.add_constant(df[cols_dr], has_constant='add'))
    return df


# %%
def segment_model_testing(lst_seg, mdl_seg, df_seg, col_target, col_dt, plt_fig=True, plt_dir_out = []):
    # functionality: 
    #   test the model on the TRN, OOS and OOT data. 
    #   if len(lst_seg)>1, the model is first tested on the segmented data and the combined the results is also generated.
    # inputs: 
    #   lst_seg: the list of segments
    #   file_model_seg: the dictionary of the model filenames for each segments
    #   file_data_seg: the dictionary of the data filenames for each segments
    #   col_target: the target column
    #   col_dt: the date column
    #   mdl_fitted: the fitted model object. if given, it overrides file_model_seg.
    #   df_data_seg: the dictionary of the dataframes for each segment. if given, it overrides file_data_seg.
    # outputs:
    #   res_testing: the result dataframe
    #   the plots of trn, oos, oot backtesting results
    def model_prediction_gini(df, col_target):
        print('segment_model_testing -> model_prediction_gini')
        df_pred_actl = df[[col_target, "Pred_"+col_target]].dropna()
        auc = roc_auc_score(df_pred_actl[col_target], df_pred_actl["Pred_"+col_target])
        gini = 2*auc-1
        return gini
    
    def model_prediction_plot(df, col_target, col_dt, plt_title_seg, plt_title_test, file_out = []):
        print('segment_model_testing -> model_prediction_plot')
        df_plot = df[[col_target, "Pred_"+col_target, col_dt]].groupby(col_dt, as_index=False).mean()
        #df_plot[col_dt] = df_plot[col_dt].apply(lambda x: dt.strptime(x, '%Y-%m-%d').date())
        df_plot[col_dt] = df_plot[col_dt].apply(lambda x: x.date())
        plt.plot(df_plot[col_dt], df_plot["Pred_"+col_target])
        plt.plot(df_plot[col_dt], df_plot[col_target])
        plt.legend(["Predicted", "Actual"])
        plt.title(f'{plt_title_seg} {plt_title_test} PD Testing')
        plt.xlabel('Date')
        plt.ylabel('Default Rate')
        plt.grid(True)
        plt.xticks(rotation=45)
        # plt.show()
        if len(file_out) > 0:
            plt.savefig(file_out)

        plt.close()
        
        return 

    
    print('segment_model_testing')
    for seg in lst_seg:
        for tst in ['TRN', "OOS", "OOT"]:
            df_seg[seg][tst] = add_model_prediction(df_seg[seg][tst], mdl_seg[seg])
    if len(lst_seg)>1:
        seg_comb = "+".join(lst_seg)
        df_seg[seg_comb] = {tst: pd.concat([df_seg[seg][tst] for seg in lst_seg]) for tst in  ['TRN', "OOS", "OOT"]}
        lst_seg += [seg_comb]
    res_testing = collections.defaultdict(dict)
    for seg in lst_seg:
        for tst in ['TRN', "OOS", "OOT"]:
            res_testing[f'Gini_{tst}'][seg] = model_prediction_gini(df_seg[seg][tst], col_target)
            res_testing[f'Pred_Mean_{col_target}_{tst}'][seg] = df_seg[seg][tst]["Pred_"+col_target].mean()
            res_testing[f'Actl_Mean_{col_target}_{tst}'][seg] = df_seg[seg][tst][col_target].mean()
            if plt_fig:
                if len(plt_dir_out) > 0:
                    model_prediction_plot(df_seg[seg][tst], col_target, col_dt, plt_title_seg=seg, plt_title_test=tst, file_out = plt_dir_out + '\\'+f'{seg}' +'_' +f'{tst}'+'_predic_plot.png')    
                else:
                    model_prediction_plot(df_seg[seg][tst], col_target, col_dt, plt_title_seg=seg, plt_title_test=tst, file_out = [])
    res_testing = pd.DataFrame(res_testing)
    return res_testing

def rank_testing_result(res, gini_wgt):
    res = res.drop_duplicates()
    res['Gini_Weighted'] = res[['Gini_OOT', 'Gini_OOS', 'Gini_TRN']].apply(lambda gini: sum(x*y for x, y in zip(gini, gini_wgt)), axis=1)
    for tst in ['TRN', 'OOS', 'OOT', 'Weighted']:
        res[f'Gini_{tst}_Rank'] = res[f'Gini_{tst}'].rank(method='min', ascending=False)
        res[f'AUC_{tst}'] = (res[f'Gini_{tst}'] + 1) * 0.5
    res['Gini_Overall_Rank'] = res[[f'Gini_{tst}_Rank' for tst in ['TRN', 'OOS', 'OOT']]].mean(axis=1).rank(method='min')
    res = res.sort_values([f'Gini_{x}_Rank' for x in ['Overall', 'Weighted', 'OOT', 'OOS', 'TRN']])
    return res
# %%
def multivariate_analysis_test(file_mdl_multivar_seg, df_seg, col_target, col_dt, gini_wgt=[],file_out_seg=[], plt_dir_seg =[]):

    res_multivariate = {} 
    for seg in df_seg.keys():  
        with open(file_mdl_multivar_seg[seg], 'rb') as file:
            res_multivariate[seg] = pickle.load(file)
        res_testing, index = [], []
        set_features = set()
        for key, val in res_multivariate[seg].items():
            features = tuple(sorted(val['Features']))
            if features not in set_features:
                set_features.add(features)
                index.append(', '.join([str(len(features))]+list(features)))
                mdl_seg = {seg: val['Model']}
                if len(plt_dir_seg) > 0:
                    res_tmp = segment_model_testing([seg], mdl_seg, df_seg, col_target, col_dt, plt_fig=True, plt_dir_out = plt_dir_seg[seg] )
                else:
                    res_tmp = segment_model_testing([seg], mdl_seg, df_seg, col_target, col_dt, plt_fig=False, plt_dir_out = [] )
                res_tmp['Model_Number'] = val['Model_Number']
                res_tmp['max_p_vaue'] = val['Stat']['p-value'].max()
                res_tmp['max_VIF'] = val['Stat']['VIF'].max()
                res_tmp['all_var_coef_neg'] = all(val['Stat'].loc[val['Stat']['Feature']!='Intercept','Coefficients']<0)
                res_tmp['Feature_Count'] = len(features)
                res_tmp['Model_Var'] = ','.join(list(features))
                res_tmp =  res_tmp.reindex(columns = ['Model_Number','Feature_Count','Model_Var'] + list(res_tmp.columns[~res_tmp.columns.isin(['Model_Number','Feature_Count','Model_Var'])]))

                res_testing.append(res_tmp)
        res_testing = pd.concat(res_testing) 
        res_testing.index = index
        res_multivariate[seg]["Testing"] = rank_testing_result(res_testing, gini_wgt)
       
        if len(file_out_seg)>0:
            res_multivariate[seg]["Testing"].to_csv(file_out_seg[seg],index=False)    
    return res_multivariate



#%%################################################################################################################
#                                     Functions on HPI Index Data Construction
##################################################################################################################

def Get_HPI_Index(Data_Input, Var_YYYYMM):
    # Purpose: Generate HPI Index by merging with Different HPI indices in SQL
    # Var_YYYYYMM:  date column (YYYYMMM) to join on in addition to Loan_Number


    def Get_HPI_Moving_Average(Data, Var_Goup_List):
        # Purpose: get moving Average HPI (past 3M, 6M and 12M)

        Data.sort_values(Var_Goup_List+['YYYYMM'], inplace= True)

        New_Data = Data.astype({'Date':'datetime64[D]'})

        New_Data.set_index('Date', inplace=True)

        New_Data['HPI_Index_3M_Avg'] = New_Data.groupby(Var_Goup_List)['HPI_Index'].transform(lambda x: x.rolling('69D',min_periods=1).mean())  # 69D is abolutely between 2 and 3 months time and for sure covers 3 months including current month 

        New_Data['HPI_Index_6M_Avg'] = New_Data.groupby(Var_Goup_List)['HPI_Index'].transform(lambda x: x.rolling('168D',min_periods=1).mean())  # 168D is abolutely between 5 and 6 months time and for sure covers 6 months including current month 

        New_Data['HPI_Index_12M_Avg'] = New_Data.groupby(Var_Goup_List)['HPI_Index'].transform(lambda x: x.rolling('359D',min_periods=1).mean())  # 379D is abolutely between 11 and 12 months time and for sure covers 12 months including current month 

        return New_Data

    
    server = 'EQSQLT01\RISK'  #Note that python is case sensitive, the server name has to be capital as they actually are

    database = 'Risk_Modelling_Dataset'


    # Dwelling Type Map
    sql_Dwelling_Type_Map = f''' 
    WITH numbered AS (
    SELECT  [DwellingTypeCode]
        ,[DwellingType]
        ,[Brookfield_Dwelling]
		,[Last_Update_Date],
        ROW_NUMBER() OVER (
          PARTITION BY
         [DwellingTypeCode]
        ,[DwellingType]
        ,[Brookfield_Dwelling]
            ORDER BY
                [Last_Update_Date] DESC
        ) AS rownum
    FROM [Risk_Modelling_Dataset].[dbo].[Mapping_DwellingCodes]
    )
    Select
    [DwellingTypeCode]
            ,[DwellingType] as Dwelling_Type
            ,[Brookfield_Dwelling]
    FROM numbered
    WHERE
        rownum = 1;
        '''    
    Dwelling_Type_Map_Data = download_from_sql(server, database, sql_Dwelling_Type_Map)
    #remove leading and trailing spaces
    Dwelling_Type_Map_Data = Dwelling_Type_Map_Data.applymap(lambda x: x.strip() if isinstance(x, str) else x) 

    # Joining to get Brookfield dwelling
    Data = Data_Input.merge(right=Dwelling_Type_Map_Data[['Dwelling_Type','Brookfield_Dwelling']].drop_duplicates(), how='left', on='Dwelling_Type')


    # FSA HPI 
    sql_HPI_FSA = f'''SELECT distinct [HPI_GeoID] as FSA
    ,[Date]
    , year(Date)*100 + month(Date) as YYYYMM
    ,[HPI_Index]
    , Style as Brookfield_Dwelling
    FROM [Risk_Modelling_Dataset].[dbo].[Staging_t_Brookfield_HPI_FSA]
    where Run_ID = {run_ID}
    order by FSA, Style, YYYYMM Asc
    '''    
    HPI_FSA_Data_01 = download_from_sql(server, database, sql_HPI_FSA).astype({'YYYYMM':'int64'})

    HPI_FSA_Data_02 = Get_HPI_Moving_Average(HPI_FSA_Data_01,['FSA','Brookfield_Dwelling'])

    HPI_FSA_Data = HPI_FSA_Data_02.copy()

    # City HPI
    sql_HPI_City = f'''  SELECT distinct
    [HPI_GeoName] as City, [HPI_GeoID]
    ,[Province] as Province_Name
    ,[Style] as Brookfield_Dwelling
    ,[Date]
     , year(Date)*100 + month(Date) as YYYYMM
    ,[HPI_Index]
    FROM [Risk_Modelling_Dataset].[dbo].[Staging_t_Brookfield_HPI_City]
    where Run_ID = {run_ID}
    order by Province, City, Style, Date Asc
    '''    
    HPI_City_Data_01 = download_from_sql(server, database, sql_HPI_City).astype({'YYYYMM':'int64'})
    
    #to separate the two cities with sme name 
    HPI_City_Data_01.loc[HPI_City_Data_01['HPI_GeoID']=='2482005','City'] = HPI_City_Data_01.loc[HPI_City_Data_01['HPI_GeoID']=='2482005','City']+'G'
    HPI_City_Data_01.loc[HPI_City_Data_01['HPI_GeoID']=='2421040','City'] = HPI_City_Data_01.loc[HPI_City_Data_01['HPI_GeoID']=='2421040','City']+'J'

    sql_Province_Map = f''' select distinct Province_Name, Province
    from [Risk_Modelling_Dataset].[dbo].[Mapping_Province_Names]
    '''    
    Province_Map_Data = download_from_sql(server, database, sql_Province_Map)

    #to separate the two cities with sme name 

    Province_Map_Data['Province_Name']= Province_Map_Data['Province_Name'].str.replace('Québec','Quebec')

    HPI_City_Data_02 = Get_HPI_Moving_Average(HPI_City_Data_01,['Province_Name','City','Brookfield_Dwelling'])

    HPI_City_Data_02['Province_Name'] = HPI_City_Data_02['Province_Name'].str.replace('Québec','Quebec')

    HPI_City_Data_03 = HPI_City_Data_02.copy()

    HPI_City_Data_03['City_Upper'] = HPI_City_Data_03['City'].str.upper()

    HPI_City_Data = HPI_City_Data_03.merge(right=Province_Map_Data,
    how='left',on='Province_Name',validate='many_to_one').drop(columns='Province_Name')


    # Metro HPI
    sql_HPI_Metro = f'''  SELECT distinct
    [HPI_GeoName] as Metro_Region_BF_FMT
    ,[Style] as Brookfield_Dwelling
    ,[Date]
     , year(Date)*100 + month(Date) as YYYYMM
    ,[HPI_Index]
    FROM [Risk_Modelling_Dataset].[dbo].[Staging_t_Brookfield_HPI_Metro]
    where Run_ID = {run_ID}
    order by Metro_Region_BF_FMT, Style, Date Asc

    '''    
    HPI_Metro_Data_01 = download_from_sql(server, database, sql_HPI_Metro).astype({'YYYYMM':'int64'})

    HPI_Metro_Data_02 = Get_HPI_Moving_Average(HPI_Metro_Data_01,['Metro_Region_BF_FMT','Brookfield_Dwelling'])

    HPI_Metro_Data_02['Metro_Region_BF_FMT_Upper'] = HPI_Metro_Data_02['Metro_Region_BF_FMT'].str.upper()

    HPI_Metro_Data = HPI_Metro_Data_02


    # Province HPI
    sql_HPI_Province = f'''   SELECT distinct 
    [HPI_GeoName] as Province_Name
    ,[Style] as Brookfield_Dwelling
    ,[Date] 
    , year(Date)*100 + month(Date) as YYYYMM
    ,[HPI_Index]
    FROM [Risk_Modelling_Dataset].[dbo].[Staging_t_Brookfield_HPI_Province]
    where Run_ID = {run_ID}
    order by Province_Name, Style, Date Asc
    '''    
    HPI_Province_Data_01 = download_from_sql(server, database, sql_HPI_Province).astype({'YYYYMM':'int64'})

    HPI_Province_Data_02 =  Get_HPI_Moving_Average(HPI_Province_Data_01,['Province_Name','Brookfield_Dwelling'])

    HPI_Province_Data_02['Province_Name'] = HPI_Province_Data_02['Province_Name'].str.replace('Québec','Quebec')

    HPI_Province_Data = HPI_Province_Data_02.merge(right=Province_Map_Data,
    how='left',on='Province_Name',validate='many_to_one').drop(columns='Province_Name')



    # Merge to get different type of HPI Indices
    Data.loc[(Data.City=="L'Ange-Gardien")&(Data.FSA[0]=='G'),'City']= Data.loc[(Data.City=="L'Ange-Gardien")&(Data.FSA[0]=='G'),'City']+'G'
    Data.loc[(Data.City=="L'Ange-Gardien")&(Data.FSA[0]=='J'),'City']= Data.loc[(Data.City=="L'Ange-Gardien")&(Data.FSA[0]=='J'),'City']+'J'

    HPI_Index_Var_List = list(HPI_FSA_Data.columns[HPI_FSA_Data.columns.str.contains('HPI_Index')])

    HPI_Index_Data_01 = Data[[Col_Loan_Nbr,'SL_Date',Var_YYYYMM,'FSA','City','Metro_Region_BF_FMT','Province','Brookfield_Dwelling','Funded_Date']].merge(right = HPI_FSA_Data[['YYYYMM','FSA','Brookfield_Dwelling'] + HPI_Index_Var_List].rename(columns = {'YYYYMM':Var_YYYYMM}|{key: key+'_FSA_Dw' for key in HPI_Index_Var_List}),how='left', on =['FSA',Var_YYYYMM,'Brookfield_Dwelling'])

    HPI_Index_Data_02 = HPI_Index_Data_01.merge(right = HPI_FSA_Data.query('Brookfield_Dwelling=="0_Aggregate"')[['FSA','YYYYMM']+HPI_Index_Var_List].rename(columns = {'YYYYMM':Var_YYYYMM}|{key: key+'_FSA_Agg' for key in HPI_Index_Var_List}),how='left', on =['FSA',Var_YYYYMM])
    
    HPI_Index_Data_02['City_Upper'] = HPI_Index_Data_02['City'].str.upper()

    HPI_Index_Data_03 = HPI_Index_Data_02.merge(right = HPI_City_Data[['YYYYMM','City_Upper','Province','Brookfield_Dwelling'] + HPI_Index_Var_List].rename(columns = {'YYYYMM':Var_YYYYMM}|{key: key+'_City_Dw' for key in HPI_Index_Var_List}),how='left', on =['City_Upper','Province',Var_YYYYMM,'Brookfield_Dwelling'], validate='many_to_one')

    HPI_Index_Data_04 = HPI_Index_Data_03.merge(right = HPI_City_Data.query('Brookfield_Dwelling=="0_Aggregate"')[['YYYYMM','City_Upper','Province']+HPI_Index_Var_List].rename(columns = {'YYYYMM':Var_YYYYMM}|{key: key+'_City_Agg' for key in HPI_Index_Var_List}),how='left', on =['City_Upper','Province',Var_YYYYMM], validate='many_to_one')

    HPI_Index_Data_04['Metro_Region_BF_FMT_Upper'] = HPI_Index_Data_04['Metro_Region_BF_FMT'].str.upper()

    HPI_Index_Data_05 = HPI_Index_Data_04.merge(right = HPI_Metro_Data[['YYYYMM','Metro_Region_BF_FMT_Upper','Brookfield_Dwelling']+HPI_Index_Var_List].rename(columns = {'YYYYMM':Var_YYYYMM}|{key: key+'_Metro_Dw' for key in HPI_Index_Var_List}),how='left', on =['Metro_Region_BF_FMT_Upper',Var_YYYYMM,'Brookfield_Dwelling'], validate='many_to_one')

    HPI_Index_Data_06 = HPI_Index_Data_05.merge(right = HPI_Metro_Data.query('Brookfield_Dwelling=="0_Aggregate"')[['YYYYMM','Metro_Region_BF_FMT_Upper']+HPI_Index_Var_List].rename(columns = {'YYYYMM':Var_YYYYMM}|{key: key+'_Metro_Agg' for key in HPI_Index_Var_List}),how='left', on =['Metro_Region_BF_FMT_Upper',Var_YYYYMM], validate='many_to_one')
    
    HPI_Index_Data_07 = HPI_Index_Data_06.merge(right = HPI_Province_Data[['YYYYMM','Province','Brookfield_Dwelling']+HPI_Index_Var_List].rename(columns = {'YYYYMM':Var_YYYYMM}|{key: key+'_Prov_Dw' for key in HPI_Index_Var_List}),how='left', on =['Province',Var_YYYYMM,'Brookfield_Dwelling'], validate='many_to_one')
        
    HPI_Index_Data_08 = HPI_Index_Data_07.merge(right = HPI_Province_Data.query('Brookfield_Dwelling=="0_Aggregate"')[['YYYYMM','Province']+HPI_Index_Var_List].rename(columns = {'YYYYMM':Var_YYYYMM}|{key: key+'_Prov_Agg' for key in HPI_Index_Var_List}),how='left', on =['Province',Var_YYYYMM], validate='many_to_one')

    HPI_Index_Data = HPI_Index_Data_08
    HPI_Index_Data.loc[Data.City.isin(["L'Ange-GardienG","L'Ange-GardienJ"]),'City']="L'Ange-Gardien"
    return HPI_Index_Data



#%%################################################################################################################
#                                     Functions on Portfolio Checking
##################################################################################################################

def data_count_by_group(Data, var_group):
    # summaryize data count by group
    sumry_by_group = pd.concat([pd.DataFrame(Data.groupby(var_group).apply(lambda x: x.shape[0],),columns=['Obs #']),pd.DataFrame(Data.groupby(var_group).apply(lambda x: x[Col_Loan_Nbr].nunique()), columns=['Unique Loan #']),pd.DataFrame(Data.groupby(var_group).apply(lambda x: x[x['Twelve_Month_FWD_Default_Ind']==1].shape[0]),columns=['DF obs #']), pd.DataFrame(Data.groupby(var_group).apply(lambda x: x[x['Twelve_Month_FWD_Default_Ind']==1].Loan_Number.nunique()),columns=['Unique DF #'])],axis=1)

    return sumry_by_group

def data_count_for_list(Data_List, Data_List_Names):
    #summarize data count by list 
    df_sumry = []

    for i in range(len(Data_List)):
        Data = Data_List[i]

        Data_Name = Data_List_Names[i]

        sumry_1 = [[Data_Name, Data.shape[0], Data[Col_Loan_Nbr].nunique(), Data[Data['Twelve_Month_FWD_Default_Ind']==1].shape[0], Data[Data['Twelve_Month_FWD_Default_Ind']==1].Loan_Number.nunique(), Data['SL_Date'].min(), Data['SL_Date'].max()]]

        df_sumry_1 = pd.DataFrame(sumry_1, columns=['Data Set','Obs #', 'Unique Loan #','DF obs #', 'Unique DF #' ,'Min Data','Max Date'] )

        df_sumry.append(df_sumry_1)
        
    return pd.concat(df_sumry)




def SFR_portfolio_overview(df,file_out):

    Balance_var = 'RemainingPrincipal_Excl_Partner'

    # Group by Alt/Prime and Sub Product, calculate data accounts, Balance (exccluding partner), % of Accounts/Balance and Total
    Pflo_Summary_01 = df.groupby(['Alt_Prime_Indicator', 'Sub_Product','Insured_Ind']).agg({Col_Loan_Nbr:'count',Balance_var:'sum'})

    Pflo_Summary_01.rename(columns={Col_Loan_Nbr:'Count', Balance_var: 'Balance (excl. partner)'}, inplace = True)

    Pflo_Summary_02 = Pflo_Summary_01.reset_index()

    Pflo_Summary_02['% of Count'] = Pflo_Summary_02['Count']/Pflo_Summary_02['Count'].sum()

    #Add total
    Pflo_Summary_02.loc['Total'] = Pflo_Summary_02.sum(numeric_only = True)

    #format columns
    Pflo_Summary_02['Balance (excl. partner)'] = Pflo_Summary_02['Balance (excl. partner)'].map('{:,.0f}'.format)

    Pflo_Summary_02['Count'] = Pflo_Summary_02['Count'].map('{:,.0f}'.format)

    Pflo_Summary_02['% of Count']  = Pflo_Summary_02['% of Count'].map('{:.1%}'.format)

    Pflo_Summary = Pflo_Summary_02
    if file_out:

        Pflo_Summary.to_excel(file_out)

    return Pflo_Summary








def hist_default_plot(Data, Seg_Var_List =[], end_date = '', Date_Column = 'SL_Date', fig_out = None):
  # purpose: plot historical default rates
    # Example Inputs:  
    # Data = SFR_PD_Data_Cleaned
    # Seg_Var_List = ['Sub_Product']
    # Seg_Var_List = ['Insured_Ind']
    # Seg_Var_List = ['Alt_Prime_Indicator']
    # Seg_Var_List = []


    # summarized data by SL_Date, sum of data counts, sum of 12mth fwd defaults

   
    Data_Label= 'Recalibration Data'

    if end_date:
        Data = Data[Data[Date_Column] <= f'{end_date}']

    DF_Hist = Data.groupby(Seg_Var_List+[Date_Column]).agg({Col_Loan_Nbr:'count','Twelve_Month_FWD_Default_Ind':'sum'})

    DF_Hist.rename(columns={Col_Loan_Nbr:'Tot_Loan_Cnt', 'Twelve_Month_FWD_Default_Ind':'DF_Cnt'},inplace=True)

    DF_Hist.reset_index(inplace=True)

    # Calculated default rates by SL_Date

    DF_Hist['Annual_DF_Rate'] = DF_Hist['DF_Cnt']/DF_Hist['Tot_Loan_Cnt']

    #DF_Hist['Annual_DF_Rate'] = DF_Hist['Annual_DF_Rate'].map('{:.1%}'.format)

    DF_Hist['New_Index'] = DF_Hist[Seg_Var_List].apply(lambda x: '_'.join(str(e) for e in list(x)),axis=1)

    New_Index_List = list(DF_Hist['New_Index'].unique())

    # Plot historical default rate

    Color_list = [ "red", "blue", "green", "yellow", "purple", "orange", "black" ]

    #fig, ax = plt.subplots(1,figsize=(5,3))
    #fig.autofmt_xdate()

    for i in range(len(New_Index_List)):
        DF_Plot = DF_Hist.loc[DF_Hist.New_Index==New_Index_List[i],:]

        # Exclude Street Capital due to abnormally high default rates due to very small data counts
        #if New_Index_List[i] != 'Street Capital':

        plt.plot(DF_Plot[Date_Column], DF_Plot['Annual_DF_Rate'], linestyle = '-', color = Color_list[i],label= New_Index_List[i])

    plt.title(f'{Data_Label}'+': Historical Annual Default Rates')
    plt.xlabel(Date_Column)
    plt.ylabel('Annaul Default Rate')
    plt.legend(title = '_'.join(Seg_Var_List),ncol=3, loc = 'best')
    if fig_out is not None:
        plt.savefig(fig_out)
    plt.close()
    # plt.show()

#%%################################################################################################################
#                                     Functions on Data Sampling
##################################################################################################################

def get_mthEnd(date):
    return date.replace(day=1)+relativedelta(months=1)-relativedelta(days=1)

# %%
def data_sampling_annually(df, col_id, col_dft, col_dt, data_freq='monthly'):
    # functionality: 
    #   annualize the data, i.e. conduct the stratified annually sampling for default and non-default leases, respectively. 
    # inputs: 
    #   col_id: the id column of the monthly snapshot data
    #   col_dft: the default indicator column of the monthly snapshot data
    #   col_dt: the date column of the monthly snapshot data
    #   data_freq: the data frequency, values in {'monthly', 'quarterly'}
    # outputs:
    #   df_res: the annualized monthly snapshots
    tail_freq = {'monthly': 12, 'quarterly': 4}.get(data_freq, None)
    if tail_freq is None:
        print("Error: Wrong data_freq!!!")
        return None
    df.loc[:,'Tmp_Month'] = df[col_dt].copy().apply(lambda x: x.month)
    df_interst = df[[col_id, col_dft, col_dt, 'Tmp_Month']]
    df_interst = df_interst.sort_values([col_id, col_dt])
    seed_rand = 1
    df_dft = df_interst[df_interst[col_dft]==1].groupby(col_id).sample(1, random_state=seed_rand)
    df_prf = df_interst[~df_interst[col_id].isin(df_dft[col_id].drop_duplicates())]
    df_prf = df_prf.groupby(col_id).tail(tail_freq)
    df_prf = df_prf.groupby(col_id).sample(1, random_state=seed_rand+1)
    df_sample = pd.concat([df_dft, df_prf])
    df_res = df.merge(df_sample[[col_id, 'Tmp_Month']], on=[col_id, 'Tmp_Month'])
    df_res = df_res.drop(columns=['Tmp_Month'])
    return df_res



# %%
def data_sampling(df,  col_id, col_dt, col_default, data_freq='monthly'):
    # functionality: 
    #   annualize the monthly snapshot data
    # inputs: 
    #   df: the to-be-sampled monthly snapshot data
    #   data_freq: the data frequency, values in {'monthly', 'quarterly'}
    #   col_dt: the date column name
    # outputs:
    #   df_res: the annualized monthly snapshots
    print('*data_sampling*')
    df.loc[:,col_dt] = df[col_dt].copy().apply(lambda x: get_mthEnd(x))    
    df_res = data_sampling_annually(df, col_id, col_default, col_dt, data_freq)
    return df_res




# %%
def data_split_trn_oos_oot(df, col_id, col_dt, col_default, window_IS, window_OOT, oos_size, fileOut_TRN, fileOut_OOS, fileOut_OOT, data_freq, stratum):
    # functionality: 
    #   split the monthly snapshot data into in-sample and oot data. 
    #   the in-sample data is further annualized and split into training and oos data.
    # inputs: 
    #   df: the to-be-sampled monthly snapshot data
    #   col_dt: the date column of the monthly snapshot data
    #   window_IS: the in-sample window 
    #   window_OOT: the out-of-time window 
    #   oos_size: the out-of-sample proportion of the in-sample data
    #   fileOut_TRN: the file name of the output training dataframe 
    #   fileOut_OOS: the file name of the output oos dataframe 
    #   fileOut_OOT: the file name of the output oot dataframe 
    #   data_freq: the data frequency, values in {'monthly', 'quarterly'}
    # outputs:
    #   df_trn: the output training dataframe 
    #   df_oos: the output oos dataframe 
    #   df_oot: the output oot dataframe 
    
    def save_oot_data(df, col_dt, window_OOT, fileOut_OOT):
        print('data_split_trn_oos_oot -> save_oot_data')
        df_oot = df[(df[col_dt]>=window_OOT[0]) & (df[col_dt]<=window_OOT[1])]
        if fileOut_OOT is not None:
            df_oot.to_csv(fileOut_OOT, index=False)
        return df_oot
    
    def save_trn_oos_data(df, col_id, col_dt, window_IS, fileOut_TRN, fileOut_OOS, oos_size, data_freq):
        print('data_split_trn_oos_oot -> save_trn_oos_data')
        df_is = df[(df[col_dt]>=window_IS[0]) & (df[col_dt]<=window_IS[1])]
        df_is_sampled = data_sampling(df_is, col_id, col_dt, col_default, data_freq)
        seed_rand = 1
        df_is_sampled['New_Stratum'] = df_is_sampled[stratum].apply(lambda x: '_'.join(str(e) for e in list(x)),axis=1) 
        df_trn, df_oos = train_test_split(df_is_sampled, test_size=oos_size, random_state=seed_rand, stratify=df_is_sampled['New_Stratum'])
        df_trn.drop(columns = ['New_Stratum'], inplace = True)
        df_oos.drop(columns = ['New_Stratum'], inplace = True)
        if fileOut_TRN is not None:
            df_trn.to_csv(fileOut_TRN, index=False)
        if fileOut_OOS is not None: 
            df_oos.to_csv(fileOut_OOS, index=False)
        return df_trn, df_oos
    
    print('data_split_trn_oos_oot')
    df_oot = save_oot_data(df, col_dt, window_OOT, fileOut_OOT)
    df_trn, df_oos = save_trn_oos_data(df, col_id, col_dt, window_IS, fileOut_TRN, fileOut_OOS, oos_size, data_freq)
    return df_trn, df_oos, df_oot

#%%
def psi_cal(df_actl, df_expect, col_val, col_prob):
    df_psi = df_expect.merge(df_actl, how='left', on=col_val, suffixes=['_Expected', '_Actual'])
    df_psi.fillna(0, inplace=True)
    psi = df_psi[[col_prob+'_Actual', col_prob+'_Expected']].apply(lambda x: (x[0]-x[1]) * np.log(x[0]/x[1]), axis=1).sum()
    return psi

#%%
def test_data_representative(dict_df, col_test, fig_dir = None):
    def get_col_dist(df, col, label):
        df_woe = pd.DataFrame(df[col].value_counts(normalize=True))
        df_woe.sort_index(inplace=True)
        df_woe.index = [f'{round(x, 2)}' for x in df_woe.index]
        return df_woe

    def gen_psi_mtx(dict_df_agg, col_test):
        keys = list(dict_df_agg.keys())
        res_mtrx = [[np.nan for _ in  range(len(keys))] for _ in  range(len(keys))]
        for i in range(len(keys)):
            for j in range(len(keys)):
                res_mtrx[i][j] = psi_cal(dict_df_agg[keys[i]], dict_df_agg[keys[j]], 'index', col_test)
        res_mtrx = pd.DataFrame(res_mtrx, columns=keys, index=keys)   
        return res_mtrx, dict_df_agg
    
    dict_df_agg = {}
    df_plt = pd.DataFrame()

    for key in list(dict_df.keys()):
        dict_df_agg[key] = dict_df[key][col_test].value_counts(normalize=True).reset_index()
        df_plt_1 = get_col_dist(dict_df[key], col_test, key)
        df_plt_1.rename(columns={col_test: key}, inplace= True)
        if df_plt.empty:
            df_plt = df_plt_1
        else: 
            df_plt = df_plt.merge(df_plt_1, left_index= True, right_index=True, how='outer' )

    df_plt.plot.bar(rot=0)
    plt.xlabel('WOE')
    plt.ylabel('Dist.')
    plt.title(f'{col_test} Distribution \n' + 'vs.'.join(dict_df.keys()))
    plt.xticks(rotation=45)
    plt.grid(True)
    if fig_dir is not None:
        plt.savefig(fig_dir+'\\Driver_Representative_'+col_test+'.png')

    plt.close()
        
    return gen_psi_mtx(dict_df_agg, col_test)

#%%################################################################################################################
#                                     Functions on SQL Extraction
##################################################################################################################
#%% Downloading data from SQL

def download_from_sql(server, database, sql, file_save=None,**kwargs):
    conn = pyodbc.connect('Driver={SQL Server};'
                          f'Server={server};'
                          f'Database={database};'
                          'Trusted_Connection=yes;')
    df = pd.read_sql_query(sql, conn,**kwargs)
    conn.close()
    if file_save:
        df.to_csv(file_save, index=False)
    return df

#%%################################################################################################################
#                                     Functions on WOE
##################################################################################################################
#%%
def penalize_missing_bin(df_woe, col_bin, col_woe, val_missing, col_skip = []):
    if pd.isna(val_missing):
        cols_missing = df_woe.loc[df_woe[col_bin].isna(), 'Variable_Name'].drop_duplicates().values
    else:
        cols_missing = df_woe.loc[df_woe[col_bin]==val_missing, 'Variable_Name'].drop_duplicates().values
    for col in cols_missing:        
        if pd.isna(val_missing):
            cond_missing_bin = (df_woe['Variable_Name']==col) & (df_woe[col_bin].isna())
        else:
            cond_missing_bin = (df_woe['Variable_Name']==col) & (df_woe[col_bin]==val_missing)

        if col not in col_skip:
            df_woe.loc[cond_missing_bin, col_woe] = df_woe.loc[df_woe['Variable_Name']==col, col_woe].min()
    return df_woe
# %%
def num_woe_mapping(val, code, penalize_new = True):
    # functionality: 
    #   map the woe according to the driver's raw value.
    # if the value did not exist in training, then penalize with the worst WOE value if penalize_new = True othersiese WOE = 0 , i.e. neutral
    # inputs: 
    #   val: the driver's raw value
    #   code: code[i, 0] is the bin's lower-bound;
    #         code[i, 1] is the bin's upper-bound;
    #         code[i, 2] is the bin's assigned value, e.g. WOE, Bin, etc.;
    # outputs:
    #   the corresponding value of code[i, 2]: the bin's assigned value, e.g. WOE, Bin, etc.;
    if val!=val:
        if code[0, 0]!=code[0, 0]:
            return code[0,2]
        else:
            if penalize_new: 
                return min(code[i, 2] for i in range(len(code)))  #worst woe value
            else: 
                return 0  # neutral woe value
    for i in range(len(code)):
        if code[i,0]<val<=code[i,1]:
            return code[i,2]
        if val==np.inf:
            return code[-1,2]  
        if val==-np.inf:
            if code[0, 0]!=code[0, 0]:
                return code[1,2]
            return code[0, 2]             



# %%
def add_woe(df, fileIn_Num=None, fileIn_Cat=None, file_out=None, penalize_new=True, override_WOE = True):
    # functionality: 
    #   add the WOE columns for the candidate drivers for the monthly snapshot.
    # if the value did not exist in training, then penalize with the worst WOE value if penalize = True othersiese WOE = 0 , i.e. neutral
    # inputs: 
    #   df: the input dataframe
    #   fileIn_Num: the codebook filename of the numeric drivers
    #   fileIn_Cat: the codebook filename of the categorical drivers
    #   file_out: the file name of the updated monthly snapshot 
    # outputs:
    #   df: the updated monthly snapshot 
    def add_num_driver_woe(df, fileIn_Num):
        print('add_num_driver_woe')
        codebook = pd.read_csv(fileIn_Num, low_memory=False) 
        codebook = codebook.query('Bin!=-9999').copy()
        codebook.loc[:,'Bin_Range_LB'] = codebook['Bin_Range_LB'].apply(lambda x: -np.inf if x=='#NAME?' else x)
        codebook.loc[:,'Bin_Range_LB'] = codebook['Bin_Range_LB'].astype(float)
        cols_tmp = ['Bin_Range_LB', 'Bin_Range_UB', 'Bin_WOE']
        codebook = {var: codebook[codebook['Variable_Name']==var][cols_tmp].values for var in codebook['Variable_Name'].unique()}
        for col in df.columns:
            if col in codebook:
                df[col+'_WOE'] = df[col].fillna(value=np.nan).apply(lambda x: num_woe_mapping(x, codebook[col],penalize_new))
        return df
    def add_cat_driver_woe(df, fileIn_Cat):
        print('add_cat_driver_woe')
        try:
            codebook = pd.read_csv(fileIn_Cat, low_memory=False) 
        except pd.errors.EmptyDataError:
            return df

        codebook = {var: {str(x): y for x, y in codebook.loc[codebook['Variable_Name']==var, ['Value','WOE']].values}  for var in codebook['Variable_Name'].unique()}
        for col in df.columns:
            if col in codebook:
                df[col+'_WOE'] = df[col].astype(str).apply(lambda x: codebook[col].get(x, min(codebook[col].values()) if penalize_new else 0))  # if the value did not exist in training, then penalize with the worst WOE value if penalize = True othersiese WOE = 0 , i.e. neutral
        return df
    def override_woe(df):
        # If Beacon missing as well, then BNI will receive most punitive WOE. Otherwise keeping no change, which was data driven WOE 

        col_BNI = df.columns[df.columns.str.contains('BNI') & df.columns.str.contains('_WOE')].str.replace('_WOE','')

        
        for var in col_BNI:

            var_corresponding_Beacon = var.replace('BNI','Beacon')  #Beacon counterpart variable
            ind_var_mis = df[var].isna()
            ind_var_Beacon_mis = df[var_corresponding_Beacon].isna()
            df.loc[ind_var_mis & ind_var_Beacon_mis, var+'_WOE'] = df[var+'_WOE'].unique().min() #if both Beacon and BNI are missing, then assign worst WOE for BNI variables. THis is because BNI is usually not required when Beacon is available

        return df

    if fileIn_Num is not None:
        df = add_num_driver_woe(df, fileIn_Num)
    if fileIn_Cat is not None: 
        df = add_cat_driver_woe(df, fileIn_Cat)

    if override_WOE:
        df = override_woe(df)  #override_WOE for special cases

    if file_out is not None:
        df.to_csv(file_out, index=False)
    return df


#%%################################################################################################################
#                                     Functions on WOE Binning
##################################################################################################################

# %%
def binning_cat_woe_by_woeDiff(fileIn, col_target, min_woe_diff= min_woe_diff, min_bin_prob=0, file_out='', cols_skip=[]):
    # functionality: 
    #   group the bins of the categorical drivers according to the woe difference.
    # inputs: 
    #   fileIn: the file name of the pre-binning codebook of the categorical drivers 
    #   col_target: the target column
    #   min_woe_diff: the minimum woe difference for binning
    #   min_bin_prob: the minimum relative bin size 
    #   file_out: the file name of the output codebook
    #   cols_skip: the to-be-skipped columns
    # outputs:
    #   df: the post-binning codebook of the categorical drivers 
    def refine_cat_bins_size(df, tot_event_na_addon, tot_nonevent_na_addon):
        df_bin = df[[col_target, 'CNT', 'Bin']].groupby('Bin', as_index=False).sum()
        df_bin = cal_add_cat_driver_woe(df_bin, 'Bin', col_target, tot_event_na_addon, tot_nonevent_na_addon)
        df_bin = df_bin.sort_values('Bin_WOE')
        df_bin['Bin_Prob'] = df_bin['CNT']/df_bin['CNT'].sum()
        if df_bin['Bin_Prob'].min()>=min_bin_prob:
            df = df.merge(df_bin[['Bin', 'Bin_WOE']], how='left', on='Bin')  
            df = df.drop(columns=['WOE'])
            df = df.rename(columns={'Bin_WOE': 'WOE'})
            df = df.sort_values(["WOE", 'Bin'])
            bins_new = list(df['Bin'].drop_duplicates().values)
            bin_map = {x: y for x ,y in zip(bins_new, range(len(bins_new)))}
            df['Bin'] = df['Bin'].map(bin_map)
            return df[['Variable_Name', 'Value', 'WOE', col_target, 'CNT', 'Bin']]
        df_bin['New_Bin'] = range(len(df_bin))
        for i in range(len(df_bin)):
            prev = abs(df_bin['Bin_WOE'].iloc[i]-df_bin['Bin_WOE'].iloc[i-1]) if i!=0 else np.nan
            nxt = abs(df_bin['Bin_WOE'].iloc[i]-df_bin['Bin_WOE'].iloc[i+1]) if i!=len(df_bin)-1 else np.nan
            if df_bin['Bin_Prob'].iloc[i]<min_bin_prob:
                if prev<=nxt or i==len(df_bin)-1:
                    df_bin['New_Bin'].iloc[i] = df_bin['New_Bin'].iloc[i-1]  
                else: 
                    df_bin['New_Bin'].iloc[i] = df_bin['New_Bin'].iloc[i+1]
                break
        df = df.merge(df_bin[['Bin', 'New_Bin', 'Bin_WOE']], how='left', on='Bin')    
        df = df.drop(columns=['Bin', 'WOE'])
        df = df.rename(columns={'Bin_WOE': 'WOE', 'New_Bin': 'Bin'})
        df = df.sort_values(["WOE", 'Bin'])
        bins_new = list(df['Bin'].drop_duplicates().values)
        bin_map = {x: y for x ,y in zip(bins_new, range(len(bins_new)))}
        df['Bin'] = df['Bin'].map(bin_map)
        return refine_cat_bins_size(df, tot_event_na_addon, tot_nonevent_na_addon)    
    def refine_cat_bins(df, tot_event_na_addon, tot_nonevent_na_addon):
        df_bin = df[[col_target, 'CNT', 'Bin']].groupby('Bin', as_index=False).sum()
        df_bin = cal_add_cat_driver_woe(df_bin, 'Bin', col_target, tot_event_na_addon, tot_nonevent_na_addon)
        num_bin = 0
        df_bin['New_Bin'] = 0
        for i in range(1, len(df_bin)):
            prev = abs(df_bin['Bin_WOE'].iloc[i]-df_bin['Bin_WOE'].iloc[i-1])
            nxt = abs(df_bin['Bin_WOE'].iloc[i]-df_bin['Bin_WOE'].iloc[i+1]) if i!=len(df_bin)-1 else np.nan
            if nxt<prev or prev>min_woe_diff:
                num_bin += 1
            df_bin['New_Bin'].iloc[i] = num_bin  
        if num_bin==0:
            return df[['Variable_Name', 'Value', 'WOE', col_target, 'CNT', 'Bin']]
        df = df.merge(df_bin[['Bin', 'New_Bin', 'Bin_WOE']], how='left', on='Bin')    
        df = df.drop(columns=['Bin', 'WOE'])
        df = df.rename(columns={'Bin_WOE': 'WOE', 'New_Bin': 'Bin'})
        df = df.sort_values(["WOE", 'Bin'])
        bins_new = list(df['Bin'].drop_duplicates().values)
        bin_map = {x: y for x ,y in zip(bins_new, range(len(bins_new)))}
        df['Bin'] = df['Bin'].map(bin_map)
        if all(df_bin['Bin']==df_bin['New_Bin']):
            return df[['Variable_Name', 'Value', 'WOE', col_target, 'CNT', 'Bin']]
        else:
            return refine_cat_bins(df, tot_event_na_addon, tot_nonevent_na_addon)
        
    df_all = pd.read_csv(fileIn, low_memory=False)
    res = []

    if not df_all.empty: 
        for var in df_all['Variable_Name'].drop_duplicates().values[:]:
            print('binning_cat_woe_by_woeDiff:', var)
            df_var = df_all[df_all['Variable_Name']==var]
            df_na = df_var[df_all['Value'].isna()]
            if len(df_na)!=0:
                df_na['Bin'] = -1
            df_var = df_var[~df_all['Value'].isna()]
            df_var['Bin'] = [i for i in range(len(df_var))] 
            if var in cols_skip:
                df_res = pd.concat([df_na, df_var])
            else:
                if df_var.empty:
                    df_res = df_na
                else:
                    if len(df_na)>0:
                        tot_event_na_addon = df_na[col_target].iloc[0]
                        tot_nonevent_na_addon = df_na['CNT'].iloc[0]-tot_event_na_addon
                    else:
                        tot_event_na_addon = 0
                        tot_nonevent_na_addon =0
                    df_var = refine_cat_bins_size(df_var, tot_event_na_addon, tot_nonevent_na_addon)
                    df_res = pd.concat([df_na, refine_cat_bins(df_var, tot_event_na_addon, tot_nonevent_na_addon)])
            res.append(df_res)
        res = pd.concat(res)
    else:
        res = pd.DataFrame()

    if file_out:
        res.to_csv(file_out, index=False)
    return res


# %%
def gen_woe_categorical_drivers(df, file_out, col_target, cols_skip=[]):
    # functionality: 
    #   generate the woe transformation codebook for the categorical drivers without binning.
    # inputs: 
    #   df: the input dataframe
    #   file_out: the file name of the output codebook
    #   col_target: the target column
    #   cols_skip: the to-be-skipped columns
    # outputs:
    #   res_woe_summary: the pre-binning woe transformation codebook of the categorical drivers 
    res_woe_summary = []
    cols_all, cols_num, cols_cat = num_cat_col_split(df, cols_skip)
    for col_dr in cols_cat[:]:
        df_test = df[[col_dr, col_target]].copy()
        df_res = cal_add_cat_driver_woe(df_test, col_dr, col_target)
        df_res = df_res[[col_dr, col_dr+'_WOE']].drop_duplicates()
        df_res['Variable_Name'] = col_dr
        df_res = df_res.rename(columns={col_dr: 'Value', col_dr+'_WOE': 'WOE'})
        df_res = df_res.reindex(columns=['Variable_Name', 'Value', 'WOE'])
        df_test.loc[:,'CNT'] = 1
        df_grp = df_test.groupby([col_dr], as_index=False, dropna=False).sum()
        df_grp = df_grp.rename(columns={col_dr: 'Value'})
        df_res = df_res.merge(df_grp, how='left', on='Value')
        res_woe_summary.append(df_res)
    if bool(res_woe_summary):
        res_woe_summary = pd.concat(res_woe_summary)
        res_woe_summary = res_woe_summary.sort_values(['Variable_Name', 'WOE'])
    else:
        res_woe_summary = pd.DataFrame()
    
    if file_out:
        res_woe_summary.to_csv(file_out, index=False)
    return res_woe_summary






# %%
def cal_add_cat_driver_woe(df, col_dr, col_target, tot_event_na_addon=0, tot_nonevent_na_addon=0):
    # functionality: 
    #   calculate and add the woe for the categorical driver.
    # inputs: 
    #   df: the input dataframe
    #   col_dr: the driver column
    #   col_target: the target column
    # outputs:
    #   df: the updated dataframe 
    import sys
    df_test = df[[col_dr, col_target]].copy()
    if 'CNT' not in df.columns:
        df_test.loc[:,'CNT'] = 1
    else:
        df_test.loc[:,'CNT'] = df['CNT'].copy()
    df_grp = df_test.groupby([col_dr], as_index=False, dropna=False).sum()
    df_grp['Non_Event'] = df_grp['CNT']-df_grp[col_target]
    tot_event = df_grp[col_target].sum() + tot_event_na_addon
    tot_nonEvent = df_grp['Non_Event'].sum() + tot_nonevent_na_addon
    df_grp['Event_Rate'] = df_grp[col_target].apply(lambda x: max(sys.float_info.min, x)/max(sys.float_info.min,tot_event)) #In the past, we used 0.5 to floor the counts. However in cases where both event and non-event are small counts while population is big, with one of them being 0 can cause problems. One example experiened is that 0 defaults in a category can still result in negative WOE, which is counter intuitive. THerefore, 0.5 even looking small is still not sufficiently small to result in intuitive results sometimse. Instead, we use system smallest number and later floor and cap in the steps below to avoid extreme outliers distorting logistic regression results using large WOE values.
    df_grp['Non_Event_Rate'] = df_grp['Non_Event'].apply(lambda x: max(sys.float_info.min, x)/max(sys.float_info.min, tot_nonEvent))
    df_grp[f'{col_dr}_WOE'] = df_grp[['Event_Rate', 'Non_Event_Rate']].apply(lambda x: min(5, max(-5, np.log(x[1]/x[0]))), axis=1)   # bound WOE values by -5 and 5 to avoid extreme outliers. This is already large boundaries from what is seen in experiences. Also, WOE = 5 implies log-odds of 5 times, or odds of exp(5) which is ~150 times, hence a very large number. We also need review final WOE results being aware that -5 and 5 implies large odds ratio. No big portions of the data are expected to have this large WOE (needs investigation if that happens)
    df = df.merge(df_grp[[col_dr, f'{col_dr}_WOE']], on=col_dr)
    return df
# %%  Latest update: GZ 20230930 updating with an option parameter part_opt to allow for manual binning
def woe_binning_numericVar(df, col_dr, col_target, n_max_bins, n_ini_cuts=None, min_bin_pd=0.0003, alpha=0.15, min_bin_prob=0.05, min_woe_diff= min_woe_diff, part_opt = None):
    # functionality: 
    #   optimize the partition of the value of the numeric drivers for the woe binning 
    # inputs: 
    #   df: the input dataframe
    #   col_dr: the driver column 
    #   col_target: the target column
    #   n_max_bins: the maximum number of bins
    #   n_ini_cuts: the initial number of cutoff points
    #   min_bin_pd: the minimum pd of the resulting bins
    #   alpha: the significance level when determining the monotonicity of the bins' pds. Where the PD has to be monotonic and the PD difference has to be bounded away from 0 (above or below based on monotonicity) based on assumed distributions (hence the alpha quantile)
    #   min_bin_prob: the minimum relative bin size 
    #   min_woe_diff: the minimum woe difference of the adjacent bins
    #   if part_opt is not None, then an optimal partition or manual partition is imposed. The automatic binning will be be ran. Rather, the results taken from the part_opt
    # outputs:
    #   woe_summary: the woe transformation codebook dataframe of the numeric drivers 
    print('woe_binning_numericVar:', col_dr)
    def get_cutoff_candidates(ss, n_ini_cuts):
        if n_ini_cuts is not None:
            quantile = [i/n_ini_cuts for i in range(1, n_ini_cuts)]
            cut_cand = np.unique([ss.quantile(q, interpolation='lower') for q in quantile])
            cut_cand = np.sort(cut_cand)
        else:
            cut_cand = np.sort(np.unique(ss.dropna().values))[:-1]
        return cut_cand
    def process_dr_target_df(df, col_dr, col_target, partition):
        def assign_bin_4_continuous_var(val, partition_sorted):
            if val==np.inf:
                return len(partition_sorted)-1
            if val==-np.inf:
                return 0
            for i in range(len(partition_sorted)-1):
                if partition_sorted[i]<val<=partition_sorted[i+1]:
                    return i
            return -1
        df_data = df[[col_dr, col_target]].copy()
        df_data.loc[:,'Bin'] = df_data[col_dr].apply(lambda x: assign_bin_4_continuous_var(x, partition))
        df_data.loc[:,'CNT'] = 1
        return df_data
    def valid_partition_distribution_check(df_data):
        def check_ge_min_bin_pd(df_grp, min_bin_pd):
            ss_test = df_grp.loc[df_grp['Bin']>=0, 'PD']
            return ss_test.min()>=min_bin_pd
        def check_ge_min_bin_prob(df_grp, min_bin_prob):
            ss_test = df_grp.loc[df_grp['Bin']>=0, 'CNT']
            ss_test = ss_test/df_grp['CNT'].sum()
            return ss_test.min()>=min_bin_prob
        def check_monotonic_bin_pd(df_grp, alpha):
            cv = norm.ppf(1-alpha)
            df_test = df_grp.loc[df_grp['Bin']>=0, ['Bin', 'PD', 'CNT']] 
            df_test['PD_Lag1'] = df_test['PD'].shift(1)
            df_test['CNT_Lag1'] = df_test['CNT'].shift(1)
            df_test['SE'] = df_test[['PD', 'CNT', 'PD_Lag1', 'CNT_Lag1']].apply(lambda x: math.sqrt(x[0]*(1-x[0])/x[1]+x[2]*(1-x[2])/x[3]), axis=1)
            df_test['Delta_PD'] = df_grp['PD']-df_test['PD_Lag1']
            df_test['Delta_PD_LB'] = df_test['Delta_PD']-cv*df_test['SE']
            df_test['Delta_PD_UB'] = df_test['Delta_PD']+cv*df_test['SE']
            df_test['Statistical_Desc'] = df_test['Delta_PD_UB']<0
            df_test['Statistical_Aesc'] = df_test['Delta_PD_LB']>0
            df_test = df_test.dropna()
            return df_test['Statistical_Desc'].all() or df_test['Statistical_Aesc'].all()
        df_grp = df_data.groupby('Bin', as_index=False).sum()
        df_grp['PD'] = df_grp[col_target]/df_grp['CNT']
        res = check_ge_min_bin_pd(df_grp, min_bin_pd) and check_ge_min_bin_prob(df_grp, min_bin_prob) and check_monotonic_bin_pd(df_grp, alpha)
        return res
    def valid_partition_woe_check(df_data):
        def check_monotonic_woe(df_woe):
            ss_test = df_woe['Bin_WOE']
            return ss_test.is_monotonic or ss_test.is_monotonic_decreasing
        def check_ge_min_woe_diff(df_woe):
            if len(df_woe)<=1: return True
            df_woe['Bin_WOE_Diff'] = df_woe['Bin_WOE']-df_woe['Bin_WOE'].shift(1)
            return df_woe['Bin_WOE_Diff'].abs().min()>=min_woe_diff
        df_woe = df_data[['Bin', 'Bin_WOE']].drop_duplicates()
        df_woe = df_woe.sort_values('Bin')
        df_woe = df_woe[df_woe['Bin']>=0]
        res = check_monotonic_woe(df_woe) and check_ge_min_woe_diff(df_woe)    
        return res  
    def org_opt_partition_woe(woe_opt, gini_opt, part_opt):
        woe_opt_summary = woe_opt.groupby(['Bin', 'Bin_WOE'], as_index=False).sum()
        woe_opt_summary['Bin_PD'] = woe_opt_summary[col_target]/woe_opt_summary['CNT']
        woe_opt_summary['Bin_Prob'] = woe_opt_summary['CNT']/woe_opt_summary['CNT'].sum()
        woe_opt_summary['Bin_Range'] = woe_opt_summary['Bin'].apply(lambda x: f'({round(part_opt[x],6)}, {round(part_opt[x+1],6)}]' if x!=-1 else 'Missing')
        woe_opt_summary['Bin_Range_LB'] = woe_opt_summary['Bin'].apply(lambda x: part_opt[x] if x!=-1 else np.nan)
        woe_opt_summary['Bin_Range_UB'] = woe_opt_summary['Bin'].apply(lambda x: part_opt[x+1] if x!=-1 else np.nan)
        woe_opt_summary['Variable_Name'] = col_dr
        woe_opt_summary['Group_Count'] = woe_opt_summary['CNT']
        woe_opt_summary['Default_Count'] = woe_opt_summary[col_target]
        woe_opt_summary = woe_opt_summary.reindex(columns=['Variable_Name', 'Bin', 'Bin_Range', 'Bin_WOE', 
                                                           'Bin_PD', 'Group_Count', 'Default_Count', 'Bin_Prob', 
                                                           'Bin_Range_LB', 'Bin_Range_UB'])
        woe_opt_summary['Gini'] = gini_opt
        woe_opt_summary = woe_opt_summary.sort_values('Bin')
        return woe_opt_summary
    
    if part_opt is None:
        cut_cand = get_cutoff_candidates(df[col_dr].drop_duplicates(), n_ini_cuts)
        if len(cut_cand)<n_ini_cuts-1:
            cut_cand = get_cutoff_candidates(df[col_dr].drop_duplicates(), None)
        if len(df[col_dr].unique())<=2:
            min_bin_prob = 0
        gini_opt, woe_opt = -1, None
        part_opt = [-np.inf, np.inf]
        for i in range(n_max_bins):
            print(col_dr, ':', i+1, '/', n_max_bins)
            cut_opt = None
            for cut in cut_cand:
                part_crr = sorted(np.unique(part_opt+[cut]))
                df_data = process_dr_target_df(df, col_dr, col_target, part_crr)
                if valid_partition_distribution_check(df_data):
                    df_data = cal_add_cat_driver_woe(df_data, 'Bin', col_target)
                    if valid_partition_woe_check(df_data):
                        gini = univariate_gini_auc_ks(df_data, 'Bin_WOE', col_target)['Gini'].iloc[0]
                        # gini = univariate_gini(df[col_target], df_data['bin_WOE'])
                        if gini>gini_opt:
                            gini_opt, cut_opt = gini, cut
                            woe_opt = df_data[[col_dr, col_target, 'CNT', 'Bin', 'Bin_WOE']]                  
            if cut_opt is not None:
                part_opt = sorted(np.unique(part_opt+[cut_opt]))
                print('part_opt:', part_opt, 'Max Gini:', gini_opt)
            else:
                print(col_dr, ':', i+1, '/', n_max_bins, 'No Improvement Break!!!')
                break
    else:
        part_crr = part_opt
        df_data = process_dr_target_df(df, col_dr, col_target, part_crr)
        df_data = cal_add_cat_driver_woe(df_data, 'Bin', col_target)
        gini_opt = univariate_gini_auc_ks(df_data, 'Bin_WOE', col_target)['Gini'].iloc[0]
        woe_opt = df_data[[col_dr, col_target, 'CNT', 'Bin', 'Bin_WOE']]                  

    if woe_opt is not None:    
        woe_summary = org_opt_partition_woe(woe_opt, gini_opt, part_opt) 
    else:
        columns = ['Variable_Name', 'Bin', 'Bin_Range', 'Bin_WOE', 'Bin_PD', 'Bin_Prob', 'Bin_Range_LB', 'Bin_Range_UB', 'Gini']
        woe_summary = pd.DataFrame([[col_dr]+[-9999]*(len(columns)-1)], columns=columns)
    return woe_summary


def gen_woe_binning_numeric_drivers(df, file_out, col_target, cols_skip=[], n_max_bins=20, n_ini_cuts=100, 
                                     min_bin_pd=0.0003, alpha=0.15, min_bin_prob=0.05, min_woe_diff= min_woe_diff, cols_corp=[], cols_indi=[]):
    # functionality: 
    #   generate the woe transformation codebook for the numeric drivers.
    # inputs: 
    #   df: the input dataframe
    #   file_out: the file name of the output codebook
    #   col_target: the target column
    #   cols_skip: the to-be-skipped columns
    #   n_max_bins: the maximum number of bins
    #   n_ini_cuts: the initial number of cutoff points
    #   min_bin_pd: the minimum pd of the resulting bins
    #   alpha: the significance level when determining the monotonicity of the bins' pds
    #   min_bin_prob: the minimum relative bin size 
    #   min_woe_diff: the minimum woe difference of the adjacent bins
    # outputs:
    #   res_woe_summary: the woe transformation codebook of the numeric drivers 
    res_woe_summary = []
    cols_all, cols_num, cols_cat = num_cat_col_split(df, cols_skip)
    for col_dr in cols_num:
        if col_dr in cols_corp:
            df_sp = df.query('Corporate_Indicator=="Yes"')
        elif col_dr in cols_indi:
            df_sp = df.query('Corporate_Indicator=="No"')
        else:
            df_sp = df
        woe_summary = woe_binning_numericVar(df_sp, col_dr, col_target, n_max_bins, n_ini_cuts, min_bin_pd, alpha, min_bin_prob, min_woe_diff)   
        res_woe_summary.append(woe_summary)
    if bool(res_woe_summary):
        res_woe_summary = pd.concat(res_woe_summary, axis=0)
    else: 
        res_woe_summary = pd.DataFrame()
    if file_out:
       res_woe_summary.to_csv(file_out, index=False)
    return res_woe_summary





#%%################################################################################################################
#                                     Functions on Univariate Analysis
##################################################################################################################

#%% Univariate Analysis
# %%
def unique_value_freq(df, col):
    # functionality: 
    #   count the number of unique values of the insterested column in the dataframe 
    # inputs: 
    #   df: the input dataframe
    #   col: the insterested column
    # outputs:
    #   res: the result dictionary
    res = {}
    df_val_cnt = df[[col]].copy()
    df_val_cnt['CNT'] = 1
    freq = df_val_cnt.groupby(col, as_index=False).count().sort_values('CNT', ascending=False)
    res['Count_of_Unique_Value'] = len(freq)
    if len(freq)==0:
        res['Most_Freq_Value'], res['Most_Freq_Value_Frequency'] = 0, 0
    else:
        res['Most_Freq_Value'] = freq[col].iloc[0] 
        res['Most_Freq_Value_Frequency'] = freq['CNT'].iloc[0]/len(df)
    return res

# %% univariate statistical Analysis
def univariate_stat_analysis(df, col):
    # functionality: 
    #   conduct the univariate statistical analysis 
    # inputs: 
    #   df: the input dataframe
    #   col: the insterested column
    # outputs:
    #   res: the result dataframe
    ss = df[col]
    quantiles = [0.01, 0.05, 0.25,  0.5, 0.75, 0.95, 0.99]
    res = {'Variable_Name': col}
    if str(ss.dtype) in ('int64', 'float64'):
        res['Variable_Type'] = 'Numeric'
        res['Mean'] = ss.mean()
        res['STD'] = ss.std()
        res['Skewness'] = ss.skew()
        res['Kurtosis'] = ss.kurtosis()
        for quantile in quantiles:
            pct = f'Pct_0{int(quantile*100)}' if quantile in (0.01, 0.05) else f'Pct_{int(quantile*100)}'
            res[pct] = ss.quantile(quantile)
        res['Min'] = ss.min()
        res['Max'] = ss.max()
    else:
        res['Variable_Type'] = 'Categorical'
        keys = ['Mean', 'STD', 'Skewness', 'Kurtosis'] 
        keys += [f'Pct_0{int(quantile*100)}' if quantile in (0.01, 0.05) else f'Pct_{int(quantile*100)}' for quantile in quantiles]
        keys += ['Min', 'Max']
        for key in keys:
            res[key] = np.nan
    res['Popuilation_Rate'] = len(ss[~ss.isna()])/len(df)
    res['Total_Count'] = len(ss)
    res.update(unique_value_freq(df, col))
    res = pd.DataFrame(res, index=[0])
    return res
# %% univariate statistical Analysis
def univariate_stat_analysis(df, col):
    # functionality: 
    #   conduct the univariate statistical analysis 
    # inputs: 
    #   df: the input dataframe
    #   col: the insterested column
    # outputs:
    #   res: the result dataframe
    ss = df[col]
    quantiles = [0.01, 0.05, 0.25,  0.5, 0.75, 0.95, 0.99]
    res = {'Variable_Name': col}
    if str(ss.dtype) in ('int64', 'float64'):
        res['Variable_Type'] = 'Numeric'
        res['Mean'] = ss.mean()
        res['STD'] = ss.std()
        res['Skewness'] = ss.skew()
        res['Kurtosis'] = ss.kurtosis()
        for quantile in quantiles:
            pct = f'Pct_0{int(quantile*100)}' if quantile in (0.01, 0.05) else f'Pct_{int(quantile*100)}'
            res[pct] = ss.quantile(quantile)
        res['Min'] = ss.min()
        res['Max'] = ss.max()
    else:
        res['Variable_Type'] = 'Categorical'
        keys = ['Mean', 'STD', 'Skewness', 'Kurtosis'] 
        keys += [f'Pct_0{int(quantile*100)}' if quantile in (0.01, 0.05) else f'Pct_{int(quantile*100)}' for quantile in quantiles]
        keys += ['Min', 'Max']
        for key in keys:
            res[key] = np.nan
    res['Popuilation_Rate'] = len(ss[~ss.isna()])/len(df)
    res['Total_Count'] = len(ss)
    res.update(unique_value_freq(df, col))
    res = pd.DataFrame(res, index=[0])
    return res
# %%
def univariate_categorical_distribution(df, col):
    # functionality: 
    #   obtain the categorical driver distribution in the univariate analysis 
    # inputs: 
    #   df: the input dataframe
    #   col: the insterested column
    # outputs:
    #   res: the result dataframe
    if df.columns.__contains__(col+'_WOE'):
        df_test = df[[col, col+'_WOE']].copy()
    else:    
        df_test = df[col].to_frame()   
    df_test[col] = df_test[col].fillna('Missing')
    df_test.loc[:,'Count'] = 1
    df_test['GroupDist'] = 1/len(df)
    if df.columns.__contains__(col+'_WOE'):
        dist = df_test[[col, col+'_WOE', 'Count', 'GroupDist']].groupby([col, col+'_WOE'], as_index=False).sum()
    else:
        dist = df_test[[col, 'Count', 'GroupDist']].groupby([col], as_index=False).sum()
    dist['Variable_Name'] = col
    if df.columns.__contains__(col+'_WOE'):
        dist = dist.rename(columns={col: "Value", col+'_WOE': 'WOE'})
        res = dist.reindex(columns=["Variable_Name", 'Value', 'WOE', 'Count', 'GroupDist'])
        res = res.sort_values('WOE')
    else: 
        dist = dist.rename(columns={col: "Value"})
        res = dist.reindex(columns=["Variable_Name", 'Value', 'Count', 'GroupDist'])
    return res   
#%% 
def univariate_stat_wo_outlier(df, col, test='Outer_Fence'):
    # functionality: 
    #   conduct the univariate statistical analysis with outliers removed
    # inputs: 
    #   df: the input dataframe
    #   col: the insterested column
    #   test: the approaches of outlier identification. {'X84', 'Grubbs', 'BOR'}
    # outputs:
    #   res: the result dataframe
    df_test = add_outlier_ind(df[[col]], col, test)
    df_regular = df_test.loc[df_test[f'{col}_Outlier_Ind']==False, col]
    df_outlier = df_test[df_test[f'{col}_Outlier_Ind']==True]
    if df_regular.empty:
        # this rarely occurs, should only happen when all values are 0, and hence no outliers
        res = {'Variable_Name': col,
           f'Count_of_Outliers_{test}': len(df_outlier),
           f'Prob_of_Outliers_{test}': len(df_outlier)/len(df_test),
           f'KS_Test_w/wo_{test}_Outliers': "Identical" }
    else:
        res = {'Variable_Name': col,
            f'Count_of_Outliers_{test}': len(df_outlier),
            f'Prob_of_Outliers_{test}': len(df_outlier)/len(df_test),
            f'KS_Test_w/wo_{test}_Outliers': "Identical" if kstest(df_test.loc[~df_test[f'{col}_Outlier_Ind'].isna(), col], df_regular)[1]>=0.05 else "Different"}
    # res[f'Mean{(test!="Original")*"_wo"}_{test}{(test!="Original")*"_Outliers"}'] = df_regular.mean()
    # res[f'STD{(test!="Original")*"_wo"}_{test}{(test!="Original")*"_Outliers"}'] = df_regular.std()
    # res[f'Skewness{(test!="Original")*"_wo"}_{test}{(test!="Original")*"_Outliers"}'] = df_regular.skew()
    # res[f'Kurtosis{(test!="Original")*"_wo"}_{test}{(test!="Original")*"_Outliers"}'] = df_regular.kurtosis()
    res = pd.DataFrame(res, index=[0])
    return res

# %%
def univariate_stat_wo_outlier_old_bf20231028(df, col, test='X84', k=5.2, alpha=0.95):
    # functionality: 
    #   conduct the univariate statistical analysis with outliers removed
    # inputs: 
    #   df: the input dataframe
    #   col: the insterested column
    #   test: the approaches of outlier identification. {'X84', 'Grubbs', 'BOR'}
    #   k: the multiplier for the X84 approach
    #   alpha: the parameter alpha for Grubbs approach
    # outputs:
    #   res: the result dataframe
    df_test = df[[col]].copy()
    if len(df_test[~df_test[col].isna()])>10:       
        if test=='X84':
            median = df_test[col].median()
            mad = abs(df_test[col]-median).median()
            df_test['Outlier_Ind'] = df_test[col].apply(lambda x: (x-median)>k*mad)
        elif test=='BOR':
            df_test['Outlier_Ind'] = False
        else:
            print('Wrong test!!!!')
            return
    else:
        df_test['Outlier_Ind'] = True
    df_regular = df_test.query('Outlier_Ind==False')[col]
    df_outlier = df_test.query('Outlier_Ind==True')
    res = {'Variable_Name': col}
    if test=='BOR':
        res['Total_Count'] = len(df_test)
    else:
        res[f'Count_of_Outliers_{test}'] = len(df_outlier)
    res[f'Mean_{test}'] = df_regular.mean()
    res[f'STD_{test}'] = df_regular.std()
    res[f'Skewness_{test}'] = df_regular.skew()
    res[f'Kurtosis_{test}'] = df_regular.kurtosis()
    res = pd.DataFrame(res, index=[0])
    return res
 # %%       
def univariate_numeric_corr_analysis(df, cols_num, col_target=''):
    # functionality: 
    #   conduct the correlation analysis among numeric columns
    # inputs: 
    #   df: the input dataframe
    #   cols_num: the numeric columns
    #   col_target: the target column
    # outputs:
    #   result dictionary: {'Corr_Matrix': correlation matrixes, 'Corr_Target': the correlations between the numeric columns and the target column}
    print('univariate_numeric_corr_analysis')
    df_test = df[cols_num+[col_target]] if col_target!='' else df[cols_num]
    mtx_corr = {x: df_test.corr(method=x) for x in ['pearson', 'kendall', 'spearman']}
    res_corr = []
    if col_target!='':
        for key, value in mtx_corr.items():
            corr_tmp = value.loc[value.index!=col_target, [col_target]]
            corr_tmp = corr_tmp.rename(columns={col_target: f'{key}_corr_{col_target}'})
            res_corr.append(corr_tmp)
        res_corr = pd.concat(res_corr, axis=1)
        res_corr['Variable_Name'] = res_corr.index
    return {'Corr_Matrix': mtx_corr, 'Corr_Target': res_corr}
# %%
def univariate_gini_auc_ks(df, col_dr, col_target):
    # functionality: 
    #   calculate the univariate gini, auc, ks distance and the direction.
    # inputs: 
    #   df: the input dataframe
    #   col_dr: the driver column
    #   col_target: the target column
    # outputs:
    #   res: the result dataframe
    def get_driver_target_df(df, col_dr, col_target):
        df_test = df[[col_dr, col_target]].copy()
        df_test.loc[:,'CNT'] = 1
        if str(df_test[col_dr].dtype) in ('int64', 'float64'):
            df_test = df_test.dropna().copy()
        else:
            df_test.loc[:,col_dr] = df_test[col_dr].fillna('Missing')
        return df_test
    def get_driver_grouped_df(df_test):
        df_grp = df_test.groupby(col_dr, as_index=False).sum()
        df_grp['Good'] = df_grp['CNT']-df_grp[col_target]
        tot_tgt, tot_good  = df_grp[col_target].sum(), df_grp['Good'].sum()
        df_grp['Target_Weight'] = df_grp[col_target]/tot_tgt
        df_grp['Cum_Good'] = df_grp['Good'].cumsum()
        df_grp['Cum_Good_Weight'] = df_grp['Cum_Good']/tot_good
        df_grp['Cum_Good_Weight_Lag1'] = df_grp['Cum_Good_Weight'] .shift(1).fillna(0)
        df_grp['Cum_Target'] = df_grp[col_target].cumsum()
        df_grp['Cum_Target_Weight'] = df_grp['Cum_Target']/tot_tgt
        if len(df_grp)==0: 
            df_grp['Delta_A'] = np.nan
        else:
            df_grp['Delta_A'] = df_grp[['Cum_Good_Weight', 'Cum_Good_Weight_Lag1', 'Target_Weight']].apply(
                lambda x: (x[0]+x[1])*x[2]/2, axis=1)
        df_grp['Distance'] = df_grp['Cum_Target_Weight']-df_grp['Cum_Good_Weight']
        return df_grp
    def cal_gini_auc_ks_dir(df_grp, col_dr,df_test  = pd.DataFrame()):
        # 20230705: GZ added Gini based on built-in roc_auc_score function from sklearn.metrics
        if len(df_grp)==0:
            if not(df_test.empty):
                return pd.DataFrame([[col_dr, 0, 0, 0.5, '']], columns=['Variable_Name', 'Gini', 'AUC', 'KS', 'Direction','Gini_Built_In'])
            else:
                return pd.DataFrame([[col_dr, 0, 0, 0.5, '']], columns=['Variable_Name', 'Gini', 'AUC', 'KS', 'Direction'])
        res = {'Variable_Name': col_dr}
        gini = 1-2*df_grp['Delta_A'].sum(min_count=1)
        if not(df_test.empty):
            if (df_test[col_target].nunique()==1) or (df_test[col_dr].nunique()==1):
                res['Gini_Built_In']  = np.nan  # only one target value or one driver value, cannot calculate gini
            else:
                auc_built_in = roc_auc_score(df_test[Col_Target],df_test[col_dr])
                gini_built_in = abs(2*auc_built_in - 1)
                res['Gini_Built_In'] = gini_built_in

        res['Gini'] = abs(gini)
        res['AUC'] = (res['Gini']+1)/2
        if gini<0:
            res['KS'] = abs(df_grp['Distance'].min())
            res['Direction'] = '+'
        else:
            res['KS'] = abs(df_grp['Distance'].max())
            res['Direction'] = '-'
        res = pd.DataFrame(res, index=[0])
        return res
    df_test = get_driver_target_df(df, col_dr, col_target)
    df_grp = get_driver_grouped_df(df_test)
    res = cal_gini_auc_ks_dir(df_grp, col_dr,df_test)
    return res
# %%
def univariate_woe_2_iv(df, col_target):
    # functionality: 
    #   calculate the univariate information value for the WOE transformed drivers.
    # inputs: 
    #   df: the input dataframe
    #   col_target: the target column
    # outputs:
    #   res: the result dataframe
    print('univariate_woe_2_iv')
    res = []
    for col_woe in df.columns:
        if "_WOE" in col_woe:
            df_test = df[[col_woe, col_target]].copy()
            df_test['CNT'] = 1
            df_grp = df_test.groupby([col_woe], as_index=False, dropna=False).sum()
            df_grp['Non_Event'] = df_grp['CNT']-df_grp[col_target]
            tot_event, tot_nonEvent = df_grp[col_target].sum(), df_grp['Non_Event'].sum()
            df_grp['Event_Rate'] = df_grp[col_target].apply(lambda x: max(0.5, x)/tot_event)
            df_grp['Non_Event_Rate'] = df_grp['Non_Event'].apply(lambda x: max(0.5, x)/tot_nonEvent)
            iv = ((df_grp['Non_Event_Rate']-df_grp['Event_Rate'])*df_grp[col_woe]).sum()
            res.append({'Variable_Name': col_woe, 'IV': iv})
    res = pd.DataFrame(res)
    return res
# %%
def num_cat_col_split(df, cols_skip=[]):
    # functionality: 
    #   split columns of the input dataframe into the numeric and categorical ones.
    # inputs: 
    #   df: the input dataframe
    #   cols_skip: the to-be-skipped columns
    # outputs:
    #   cols_all: all the columns
    #   cols_num: the numeric columns
    #   cols_cat: the categorical columns
    cols_all = [x for x in df.columns if x not in cols_skip]
    cols_num, cols_cat = [], []
    for col in cols_all:
        if str(df[col].dtype) in ('int64', 'float64'):
            cols_num.append(col)
        else:
            cols_cat.append(col)
    return cols_all, cols_num, cols_cat

def univariate_analysis(df, col_target ='', fileIn_num_woe = '', cols_skip=[], file_out=''):
    # functionality: 
    #   conduct the univariate analysis.
    # inputs: 
    #   df: the input dataframe
    #   col_target: the target column
    #   fileIn_num_woe: the file name of the woe binning codebook for the numeric drivers
    #   cols_skip: the to-be-skipped columns
    #   file_out: the file name of the output dataframe
    # outputs:
    #   result dictionary: {"Stats": statistical analysis result, "Category_Dist": the distribution of categorical drivers, 
    #                       'Correlation': the result of correlation analysis, 'Outlier_Analysis': the result of outlier analysis}
    def org_stat_res(df, cols_all):
        res_stat = [univariate_stat_analysis(df, col) for col in cols_all[:]]
        res_stat = pd.concat(res_stat, axis=0)
        var_b4_woe = [x for x in res_stat['Variable_Name'].values if x[-4:]!='_WOE']
        res_popRate_b4WOE = res_stat.loc[res_stat['Variable_Name'].isin(var_b4_woe), ['Variable_Name', 'Popuilation_Rate']]
        res_popRate_b4WOE = res_popRate_b4WOE.rename(columns={'Popuilation_Rate': "Popuilation_Rate_b4Woe"})
        res_popRate_b4WOE['Variable_Name'] = res_popRate_b4WOE['Variable_Name'].apply(lambda x: x+"_WOE")
        res_stat = res_stat.merge(res_popRate_b4WOE, how='left', on='Variable_Name')
        res_stat['Popuilation_Rate'] = res_stat[['Popuilation_Rate', "Popuilation_Rate_b4Woe"]].apply(
            lambda x: x[1] if x[1]==x[1] else x[0], axis=1)
        res_stat = res_stat.drop(columns=["Popuilation_Rate_b4Woe"])
        return res_stat
    def org_dist_res(df, cols_cat):
        res_dist = [univariate_categorical_distribution(df, col) for col in cols_cat[:]]
        res_dist = pd.concat(res_dist, axis=0)
        res_dist = res_dist.reset_index(drop=True)
        return res_dist
    def org_outlier_res(df, cols_num):
        res_outlier = None
        for test in ['Outer_Fence', 'Inner_Fence',  'X84']:
            res_outlier_tmp = [univariate_stat_wo_outlier(df, col, test=test) for col in cols_num[:] if col[-4:]!='_WOE']
            res_outlier_tmp = pd.concat(res_outlier_tmp, axis=0)
            if res_outlier is None:
                res_outlier = res_outlier_tmp
            else:
                res_outlier = res_outlier.merge(res_outlier_tmp, how='left', on='Variable_Name')
        return res_outlier
    def org_gini_res(df, cols_num, col_target):
        print('org_gini_res')
        res = [univariate_gini_auc_ks(df[cols_num+[col_target]], col_dr, col_target) for col_dr in cols_num]
        res = pd.concat(res, axis=0)
        df_org_dir = res.loc[~res['Variable_Name'].str.contains('_WOE'), ['Variable_Name', 'Direction']]
        df_org_dir['Variable_Name'] = df_org_dir['Variable_Name'].apply(lambda x: x+'_WOE')
        df_org_dir = df_org_dir.rename(columns={'Direction': 'Direction_before_WOE'})
        res = res.merge(df_org_dir, how='left', on='Variable_Name')
        return res
    def get_num_woe_direction():
        codebook = pd.read_csv(fileIn_num_woe, low_memory=False) 
        woe_direction = {'Variable_Name': [], 'WOE_Direction': []}
        for var in codebook['Variable_Name'].unique():
            woe_direction['Variable_Name'].append(var+'_WOE')
            df_test = codebook[(codebook['Variable_Name']==var) & (codebook['Bin']>=0)]
            if df_test['Bin_WOE'].is_monotonic:
                woe_direction['WOE_Direction'].append('Ascending')
            elif df_test['Bin_WOE'].is_monotonic_decreasing:
                woe_direction['WOE_Direction'].append('Descending')
            else:
                woe_direction['WOE_Direction'].append('No Direction')
        woe_direction = pd.DataFrame(woe_direction)
        return woe_direction
    cols_all, cols_num, cols_cat = num_cat_col_split(df, cols_skip)
    res_stat = org_stat_res(df, cols_all)
    res_dist=[]
    if cols_cat:
        res_dist = org_dist_res(df, cols_cat)
    if col_target:
        res_gini = org_gini_res(df, cols_num, col_target)
        res_stat = res_stat.merge(res_gini, how='left', on='Variable_Name')
        res_iv = univariate_woe_2_iv(df, col_target)
        
        if not(res_iv.empty):
            res_stat = res_stat.merge(res_iv, how='left', on='Variable_Name')
    if fileIn_num_woe:
        woe_direction = get_num_woe_direction()
        res_stat = res_stat.merge(woe_direction, how='left', on='Variable_Name')  
    if col_target: 
        if fileIn_num_woe:
            res_stat['Multivariate_Analysis'] = res_stat[['Popuilation_Rate', 'Gini', 'IV']].apply(lambda x: "Include" if x[0]>=0.6 and x[1]>=0.05 and x[2]>=0.1 else 'Exclude', axis=1)
        else: 
            res_stat['Multivariate_Analysis'] = res_stat[['Popuilation_Rate', 'Gini']].apply(lambda x: "Include" if x[0]>=0.6 and x[1]>=0.05 else 'Exclude', axis=1)

        res_stat = res_stat.sort_values(['Multivariate_Analysis', 'Variable_Type', 'Gini', 'Variable_Name'], ascending=[False, False, False, True])
    else:
        res_stat = res_stat.sort_values(['Variable_Type', 'Variable_Name'], ascending=[False, True])

    res_stat = res_stat.reset_index(drop=True)
    res_corr = univariate_numeric_corr_analysis(df, cols_num, col_target)
    if col_target:
        res_stat = res_stat.merge(res_corr['Corr_Target'], how='left', on='Variable_Name')    
    res_outlier = org_outlier_res(df, cols_num)
    if file_out:
        writer = pd.ExcelWriter(file_out)
        res_stat.to_excel(writer, 'Summary')
        if not res_dist.empty:
            res_dist.to_excel(writer, 'Categorical_Dist_WOE')
        for key, mtr in res_corr['Corr_Matrix'].items():
            mtr.to_excel(writer, f'CorrMatrix_{key}')
        res_outlier.to_excel(writer, 'Outlier')
        writer.save()
        writer.close()
    return {"Stats": res_stat, "Category_Dist": res_dist, 'Correlation': res_corr, 'Outlier_Analysis': res_outlier}




#%%################################################################################################################
#                                     Final Model Training and Tests
##################################################################################################################

# %%    
def model_training(df, col_target, cols_dr, mdl_file='', dir_out = []):
    # functionality: 
    #   train and save the model
    # inputs: 
    #   df: the training dataframe
    #   col_target: the target column
    #   cols_dr: the list of driver columns
    #   mdl_file: the pickle file name to trained model and the correpsonding model information
    # outputs:
    #   res: {'Stat': the model stats, 'Deciles': the predicions deciles, 'Model': the fitted model}
    ss_y = df[col_target]
    mdl_trn = sm.Logit(ss_y, sm.add_constant(df[cols_dr], has_constant='add')).fit(disp=0)
    auc = roc_auc_score(ss_y, mdl_trn.predict())
    stat_model_best = get_model_stats(mdl_trn, auc)
    res_decile = get_model_predicted_deciles(mdl_trn)
    
    res = {'Stat': stat_model_best, 'Deciles': res_decile, 'Model': mdl_trn, 'Features': cols_dr}

    if len(dir_out)>0:
        with pd.ExcelWriter(dir_out + '\\Final_Model_Sumry.xlsx') as Writer:
            stat_model_best.to_excel(Writer, sheet_name='Stat')
            res_decile.to_excel(Writer, sheet_name='Deciles')
    if mdl_file:
        with open(mdl_file, 'wb') as file:
            pickle.dump(res, file)
    return res

#%%
#################################################################################################################
#                                     Segmentation Model Functions
##################################################################################################################
def split_mrs_df_trn_oos_oot(df_sp, ls_window_IS, ls_window_OOT, oos_size=0.2,col_date = 'SL_Date'):
    # functionality: 
    #   split and add the indicator to the MRS data to separat training, oos, and oot data
    # inputs:
    #   df_sp: the snapshot data
    #   ls_window_IS: in-sample date range
    #   ls_window_OOT: oot date range
    #   oos_size: oos portion to the in-sample data
    # outputs:
    #   df_sp: the snapshot data with Split indicator added

    print('split_mrs_df_trn_oos_oot')
    df_sp = df_sp.reset_index(drop=True)
    df_sp['Split'] = df_sp[col_date].apply(lambda x: 'TRN' if dt.strptime(ls_window_IS[0],'%Y-%m-%d') <= x <= dt.strptime(ls_window_IS[1],'%Y-%m-%d') 
                                                              else('OOT' if dt.strptime(ls_window_OOT[0],'%Y-%m-%d') <= x <= dt.strptime(ls_window_OOT[1],'%Y-%m-%d') else np.nan))
    df_trn, df_oos = train_test_split(df_sp[df_sp['Split']=='TRN'], test_size=oos_size, random_state=10)
    df_sp.loc[df_oos.index, 'Split'] = 'OOS'
    return df_sp


#%%
def cal_MRS_Bin_mapped_pd(df_sp, col_dft_ind,col_date = 'SL_Date'):
    # functionality: 
    #   calculate the mapped pd for each MRS segment
    # inputs:
    #   df_sp: the snapshot dataframe 
    #   edges: the cutoff point for the segmentation model
    #   col_dft_ind: the the colum name of the default indicator 
    # outputs:
    #   dct_mapped_pd: the dictionary of mapped pd for each segment

    df_dr_4_each_bin = df_sp.groupby([col_date, 'MRS_Bin'], as_index=False)[col_dft_ind].mean()
    df_dr_4_each_bin = df_dr_4_each_bin.groupby(['MRS_Bin'])[[col_dft_ind]].mean()
    dct_mapped_pd = df_dr_4_each_bin.iloc[:,0].to_dict()

    df_dr_4_each_bin_agg = df_sp.groupby(['MRS_Bin'])[[col_dft_ind]].mean()
    dct_pd_agg = df_dr_4_each_bin_agg.iloc[:,0].to_dict()

    return dct_mapped_pd, dct_pd_agg
#%%
def binning_w_decision_tree(df_trn, col_target, max_mrs_bins, min_mrs_bin_dist):
    tree = DecisionTreeClassifier(criterion='gini', max_depth=100, max_leaf_nodes=max_mrs_bins, 
                                    min_samples_split=min_mrs_bin_dist,  min_samples_leaf=min_mrs_bin_dist,
                                    max_features=1)
    # tree = DecisionTreeClassifier(criterion='log_loss', max_depth=100, max_leaf_nodes=max_mrs_bins, 
    #                               min_samples_split=min_mrs_bin_dist,  min_samples_leaf=min_mrs_bin_dist,
    #                               max_features=1)
    tree.fit(df_trn[['Pred_'+col_target]], df_trn[[col_target]])
    edges = sorted([x for x in tree.tree_.threshold if x!=-2]+[0, 1])
    return edges

#%%
def load_pd_models(seg_model, file_scoring_mdl_seg, file_mrs_mdl_seg): 
    # functionality: 
    #   load the PD scoring and segmentation model
    # inputs:
    #   seg_model: tuple of the segments for the scoring model
    #   file_scoring_mdl_seg: the scoring model file name 
    #   file_mrs_mdl_seg: the segmentation model file name 
    # outputs:
    #   mdl_pd: dictionary of loaded PD models

    mdl_pd = collections.defaultdict(dict)
    
    with open(file_scoring_mdl_seg[seg_model], 'rb') as file:
        scoring_mdl = pickle.load(file)

    mdl_pd['Score'] = scoring_mdl['Model']
    mdl_pd['Stat'] =   scoring_mdl['Stat']

    with open(file_mrs_mdl_seg[seg_model], 'rb') as file:
        mdl_pd['MRS'] = pickle.load(file)  

    return mdl_pd

#%%
def plot_MRS_Bin_overall_dist(df_sp_mrs, col_target, col_snap_date = 'SL_Date', Fig_Dir = None, var_PD = 'Mapped_PD'):
    # functionality: 
    #   plot the overall distribution for the segment of the segmentation model 
    # inputs:
    #   df_sp_mrs: the snapshot data with MRS bins assigned
    #   col_target: the target variable of the segmentation model 
    # outputs:
    #   df_plt: the dataframe for the figure plotting

    print('plot_MRS_Bin_overall_dist')
    df_bin_dist = pd.DataFrame(df_sp_mrs['MRS_Bin'].value_counts(normalize=True, dropna=False))
    df_bin_dist.rename(columns={'MRS_Bin': 'Account Distribution'}, inplace=True)
    df_bin_dft_dist = pd.DataFrame(df_sp_mrs.query(f"{col_target}==1")['MRS_Bin'].value_counts(normalize=True, dropna=False))
    df_bin_dft_dist.rename(columns={'MRS_Bin': 'Default Distribution'}, inplace=True)
    df_plt = df_bin_dist.merge(df_bin_dft_dist, left_index=True, right_index=True)
    df_bin_mapped_pd = df_sp_mrs.groupby([col_snap_date, 'MRS_Bin'], as_index=False)[[var_PD]].mean()
    df_bin_mapped_pd = df_bin_mapped_pd.groupby(['MRS_Bin'])[[var_PD]].mean()
    df_bin_mapped_pd.rename(columns={var_PD: var_PD.replace('_', ' ')}, inplace=True)
    df_plt = df_plt.merge(df_bin_mapped_pd, left_index=True, right_index=True)
    df_bin_dr = df_sp_mrs.groupby([col_snap_date, 'MRS_Bin'], as_index=False)[[col_target]].mean()
    df_bin_dr = df_bin_dr.groupby(['MRS_Bin'])[[col_target]].mean()
    df_bin_dr.rename(columns={col_target: 'Default Rate'}, inplace=True)
    df_plt = df_plt.merge(df_bin_dr, left_index=True, right_index=True)
    df_plt.sort_index(inplace=True)
    df_plt.rename_axis('MRS_Bin',inplace=True)
    fig, ax = plt.subplots()
    width = 0.2
    ax.bar(df_plt.index-width/2, df_plt['Account Distribution'], width, label='Account Dist.', color='g')
    ax.bar(df_plt.index+width/2, df_plt['Default Distribution'], width, label='Default Dist.', color='r')
    ax.plot(df_plt.index, df_plt[var_PD.replace('_', ' ')], label=var_PD.replace('_', ' '))
    ax.plot(df_plt.index, df_plt['Default Rate'], label='Default Rate')
    ax.legend()
    ax.set_xlabel('Segment')
    ax.set_title('Segment Distribution')
    plt.grid(True)
    # plt.show()
    if Fig_Dir is not None:
        plt.savefig( Fig_Dir +'\\MRS_Bin_overall_dist.png')
    plt.close()
    return df_plt


#%%
def plot_MRS_Bin_mthly_dist(df_sp_mrs, col_snap_date = 'SL_Date', Fig_Dir = None):
    # functionality: 
    #   plot the monthly distribution for the segment of the segmentation model 
    # inputs:
    #   df_sp_mrs: the snapshot data with MRS bins assigned
    # outputs:
    #   df_plt: the dataframe for the figure plotting

    print('plot_MRS_Bin_mthly_dist')
    if df_sp_mrs[col_snap_date].dtype=='O':
        df_sp_mrs[col_snap_date] = df_sp_mrs[col_snap_date].apply(pd.to_datetime)
    # plot distribution over time
    df_plt = df_sp_mrs.groupby(col_snap_date)['MRS_Bin'].value_counts(normalize=True, dropna=False).unstack()
    df_plt.rename(columns={col: f'Seg {col}' for col in df_plt.columns}, inplace=True)
    df_plt.plot(kind='area', stacked=True).legend(loc='upper left',bbox_to_anchor=(1,1))
    if col_snap_date == 'SL_Date':
        plt.xlabel('Date')
    if col_snap_date == 'SL_Quarter':
        plt.xlabel('Quarter')
    plt.ylabel('Segment as % of population')
    plt.title('Segment Distribution by Date')
    # plt.show()
    if Fig_Dir is not None:
        if col_snap_date == 'SL_Date':
            plt.savefig( Fig_Dir +'\\MRS_Bin_mthly_dist.png',bbox_inches='tight')
        if col_snap_date == 'SL_Quarter':
            plt.savefig( Fig_Dir +'\\MRS_Bin_qtrly_dist.png',bbox_inches='tight')
    plt.close()
    # plot data count over time

    df_count_plt = df_sp_mrs.groupby(col_snap_date)['MRS_Bin'].value_counts(normalize = False, dropna=False).unstack()
    df_count_plt.rename(columns={col: f'Seg {col}' for col in df_count_plt.columns}, inplace=True)
    df_count_plt.plot(kind='area', stacked=True).legend(loc='upper left',bbox_to_anchor=(1,1))
    if col_snap_date == 'SL_Date':
        plt.xlabel('Date')
    if col_snap_date == 'SL_Quarter':
        plt.xlabel('Quarter')
    plt.ylabel('Loan Count')
    plt.title('Segment Data Count by Date')
    # plt.show()
    if Fig_Dir is not None:
        if col_snap_date == 'SL_Date':
            plt.savefig( Fig_Dir +'\\MRS_Bin_mthly_count.png',bbox_inches='tight')
        if col_snap_date == 'SL_Quarter':
            plt.savefig( Fig_Dir +'\\MRS_Bin_qtrly_count.png',bbox_inches='tight')
    plt.close()


    # plot default count over time

    df_target_sum_plt = df_sp_mrs.groupby([col_snap_date, 'MRS_Bin']).agg({Col_Target: 'sum'}).squeeze().unstack()
    df_target_sum_plt.rename(columns={col: f'Seg {col}' for col in df_target_sum_plt.columns}, inplace=True)
    df_target_sum_plt.plot(kind='area', stacked=True).legend(loc='upper left',bbox_to_anchor=(1,1))
    if col_snap_date == 'SL_Date':
        plt.xlabel('Date')
    if col_snap_date == 'SL_Quarter':
        plt.xlabel('Quarter')
    plt.ylabel('Default Count')
    plt.title('Segment Default Count by Date')
    # plt.show()
    if Fig_Dir is not None:
        if col_snap_date == 'SL_Date':
            plt.savefig( Fig_Dir +'\\MRS_Bin_mthly_default_count.png',bbox_inches='tight')
        if col_snap_date == 'SL_Quarter':
            plt.savefig( Fig_Dir +'\\MRS_Bin_qtrly_default_count.png',bbox_inches='tight')
    plt.close()


    return df_plt, df_count_plt, df_target_sum_plt


#%%
def plot_mrs_bin_dr_backtesting(df_sp_mrs, col_target, col_snap_date = 'SL_Date', Fig_Dir = None):
    # functionality: 
    #   plot the monthly DR of the each mrs bin
    # inputs:
    #   df_sp_mrs: the snapshot data with MRS bins assigned
    #   col_target: the column name of the targe variable
    #   exclude_covid: indicator for excluding covid period 
    # outputs:
    #   df_plt: the dataframe for the figure plotting

    print('plot_mrs_bin_dr_backtesting')
    if df_sp_mrs[col_snap_date].dtype=='O':
        df_sp_mrs[col_snap_date] = df_sp_mrs[col_snap_date].apply(pd.to_datetime)
    df_plt = df_sp_mrs.groupby([col_snap_date, 'MRS_Bin'])[col_target].mean().unstack()
    df_plt.rename(columns={col: f'Seg {col}' for col in df_plt.columns}, inplace=True)
    df_plt.plot(kind='line', stacked=False, logy=True).legend(loc='upper left',bbox_to_anchor=(1,1))
    if col_snap_date == 'SL_Date':
        plt.xlabel('Date')
    if col_snap_date == 'SL_Year':
        plt.xlabel('Year')
    if col_snap_date == 'SL_Quarter':
        plt.xlabel('Quarter')
    plt.ylabel('Realized Default Rate')
    plt.title('Segmentation Model Backtesting')
    # plt.show()
    if Fig_Dir is not None:
        if col_snap_date == 'SL_Date':
            plt.savefig( Fig_Dir +'\\MRS_Bin_DR_Backtesting.png',bbox_inches='tight')
        if col_snap_date == 'SL_Year':
            plt.savefig( Fig_Dir +'\\MRS_Bin_DR_Backtesting_year.png',bbox_inches='tight')
        if col_snap_date == 'SL_Quarter':
            plt.savefig( Fig_Dir +'\\MRS_Bin_DR_Backtesting_quarter.png',bbox_inches='tight')
    plt.close()
    return df_plt


#%%
def pd_backtesting_plot_ci_err_stat(df_sp_mrs, col_target, col_pred, alpha=0.05, test='z-test', tail=1, qrt_end=True, cols_seg=[], col_snap_date = 'SL_Date', Fig_Dir = None):
    # functionality: 
    #   conduct backtesing and plot the monthly DR with confidence interval.
    # inputs:
    #   df_sp_mrs: the snapshot data with MRS bins assigned
    #   col_target: the column name of the targe variable
    #   col_pred: the column name of the prediction
    #   alpha: the significance level for the confidence interval
    #   test: 'z-test' or 't-test' for the CI
    #   tail: indicator 1 or 2 tails test
    #   qrt_end: indicator whether consider quarter end data only
    #   cols_seg: columns for segment level analysis
    # outputs:
    #   summarized results

    print('pd_backtesting_plot_ci_err_stat')  
    def plot_realized_pred_dr(df_plt, seg_col=[], seg_val=[], Fig_Out = None):
        if df_plt[col_snap_date].dtype=='O':
            df_plt[col_snap_date] = df_plt[col_snap_date].apply(pd.to_datetime)
        plt.plot(df_plt[col_snap_date], df_plt[col_target])
        plt.plot(df_plt[col_snap_date], df_plt[col_pred])
        plt.plot(df_plt[col_snap_date], df_plt[col_target+'_UB'], linestyle='--')
        plt.plot(df_plt[col_snap_date], df_plt[col_target+'_LB'], linestyle='--')
        plt.legend(['Realized PD', 'Average Mapped PD', 'CI UB', 'CI LB'])
        plt.xlabel('Date')
        plt.ylabel('Realized Default Rate')
        plt.xticks(rotation=45)
        title_suffix = ', '.join(f'{col}={val}' for col, val in zip(seg_col, seg_val))
        if not bool(title_suffix):
            title_suffix = ', ' + title_suffix  
        plt.title(f'Segmentation Model Backtesting ({tst+title_suffix})')
        plt.grid(True)
        # plt.show()
        if Fig_Out is not None:
            plt.savefig(Fig_Out)
        plt.close()
        return
    def plot_realized_pred_dr_no_ci(df_plt, seg_col=[], seg_val=[], Fig_Out = None):
        if df_plt[col_snap_date].dtype=='O':
            df_plt[col_snap_date] = df_plt[col_snap_date].apply(pd.to_datetime)
        plt.plot(df_plt[col_snap_date], df_plt[col_target])
        plt.plot(df_plt[col_snap_date], df_plt[col_pred])
        plt.legend(['Realized PD', 'Average Mapped PD'])
        plt.xlabel('Date')
        plt.ylabel('Realized Default Rate')
        plt.xticks(rotation=45)
        title_suffix = ', '.join(f'{col}={val}' for col, val in zip(seg_col, seg_val))
        if not bool(title_suffix):
            title_suffix = ', ' + title_suffix  
        plt.title(f'Segmentation Model Backtesting ({tst+title_suffix})')
        plt.grid(True)
        # plt.show()
        if Fig_Out is not None:
            plt.savefig(Fig_Out)
        plt.close()
        return 
    def gather_portfolio_lvl_res(df_tst, tst, res):
        df_plt = df_tst.copy()
        if Fig_Dir is None:
            Fig_Out = None
        else:
            Fig_Out= Fig_Dir + '\\MRS_pd_backtesting_plot_ci_pflo'+f'{tst}'+'_lvl.png'
            Fig_Out_no_ci= Fig_Dir + '\\MRS_pd_backtesting_plot_pflo'+f'{tst}'+'_lvl.png'
        plot_realized_pred_dr(df_plt = df_plt, seg_col = [], seg_val = [], Fig_Out = Fig_Out)
        plot_realized_pred_dr_no_ci(df_plt = df_plt, seg_col = [], seg_val = [], Fig_Out = Fig_Out_no_ci)
        res.append([tst, mean_squared_error(df_plt[col_target], df_plt[col_pred], squared=False), (df_plt[col_pred]-df_plt[col_target]).mean(), 
                    len(df_plt[df_plt[col_target+'_LB']>df_plt[col_pred]]), len(df_plt)])
        return res
    def gether_seg_lvl_res(df_tst, tst, df_sp_mrs, cols_seg, res):
        segments = df_sp_mrs[cols_seg].drop_duplicates().values
        for val_seg in segments:
            df_plt = df_tst.copy()
            for col, val in zip(cols_seg, val_seg):
                df_plt = df_plt[df_plt[col]==val]
            if Fig_Dir is None:
                Fig_Out = None
            else:
                Fig_Out= Fig_Dir + '\\MRS_pd_backtesting_plot_ci_pflo'+f'{tst}'+'_Seg_'+'_'.join(list(str(val_seg))) +'.png'              
            plot_realized_pred_dr(df_plt = df_plt, seg_col = cols_seg , seg_val = val_seg, Fig_Out = Fig_Out)
            if val_seg in df_tst['MRS_Bin'].unique():
                res.append([tst]+list(val_seg)+[mean_squared_error(df_plt[col_target], df_plt[col_pred], squared=False), (df_plt[col_pred]-df_plt[col_target]).mean(),len(df_plt[df_plt[col_target+'_LB']>df_plt[col_pred]]), len(df_plt)]) 
            else:
                res.append([tst]+list(val_seg)+[np.nan, np.nan, 0, 0])
        return res
    def cal_ci_dr_pd(df_sp_mrs, tst, cols_seg, col_target, col_pred, test, tail, alpha, qrt_end):
        df_tst = df_sp_mrs.copy() if tst == 'All' else df_sp_mrs.query(f"Covid_Period_Flag=='Y'") if tst == 'COVID'  else df_sp_mrs.query(f"Split=='{tst}'")
        df_res = df_tst.groupby([col_snap_date]+cols_seg, as_index=False)[[col_target, col_pred]].agg({col_target: ('mean', 'count'), col_pred: 'mean'})
        df_res.columns = [col_snap_date] + cols_seg + [col_target, 'Count', col_pred]
        df_std = df_tst.groupby([col_snap_date]+cols_seg, as_index=False)[[col_target]].std()
        df_std.rename(columns={col_target: col_target+'_STD'}, inplace=True)
        df_res = df_res.merge(df_std, on=[col_snap_date]+cols_seg)
        if test=='z-test':
            df_res['Critical_Value']= norm.ppf(1-alpha/tail)
        elif test=='t-test':
            df_res['Critical_Value'] = df_res['Count'].apply(lambda x: stats.t.ppf(1-alpha/tail, x-1))
        df_res[col_target+'_LB'] = df_res[[col_target, 'Critical_Value', col_target+'_STD', 'Count']].apply(lambda x: max(0, x[0] - x[1]*x[2]/x[3]**0.5), axis=1)
        df_res[col_target+'_UB'] = df_res[[col_target, 'Critical_Value', col_target+'_STD', 'Count']].apply(lambda x: min(1, x[0] + x[1]*x[2]/x[3]**0.5), axis=1)
        if qrt_end:
            if df_res[col_snap_date].dtype=='O':
                df_res[col_snap_date] = df_res[col_snap_date].apply(pd.to_datetime)
            df_res = df_res[df_res[col_snap_date].dt.month.isin([3, 6, 9, 12])]
        return df_res
        
    res, df_tst = [], {}
    test_split = [x for x in list(df_sp_mrs['Split'].unique()) if x==x]+['All','COVID']
    test_split = sorted(test_split, key=lambda x: {tst: i for i, tst in enumerate(['TRN', 'OOS', 'OOT', 'All','COVID'])}.get(x, np.inf))
    for tst in test_split:
        df_tst[tst] = cal_ci_dr_pd(df_sp_mrs, tst, cols_seg, col_target, col_pred, test, tail, alpha, qrt_end)       
        if not bool(cols_seg):
            res = gather_portfolio_lvl_res(df_tst[tst], tst, res)
        else:
            res = gether_seg_lvl_res(df_tst[tst], tst, df_sp_mrs, cols_seg, res)         
    res = pd.DataFrame(res, columns=["Test"]+cols_seg+['RMSE', 'ME', '# Downward Breaches', 'Total # Observations'])
    res.sort_values(['Test']+cols_seg, inplace=True)
    return {'Summary': res, 'Detail': df_tst}

#%%    
def heterogeneity_test_1to1(seg_1, seg_2, alpha=0.05):
    """ performs heterogeneity test to two PD segments
    :param seg_1: pandas.series or array including the default indicator for a segment
    :param seg_2: pandas.series or array including the default indicator for a segment
    :param alpha: significance level
    :return: test_pvalue, hypothesis test results (True: H0 is rejected i.e. the segments are different)
    """
    dr_1 = np.mean(seg_1)
    dr_2 = np.mean(seg_2)
    n_1 = len(seg_1)
    n_2 = len(seg_2)
    dr = (np.sum(seg_1) + np.sum(seg_2)) / (n_1 + n_2)
    z = (dr_1 - dr_2) / np.sqrt(dr * (1 - dr) * (1 / n_1 + 1 / n_2))
    p_value = norm.cdf(-np.abs(z))
    return p_value, p_value < alpha

#%%
def heterogeneity_test_multi_seg(df, col_target, segment_id, alpha=0.05):
    """
    performs heterigeneity test for a portfolio
    :param df: dataframe containing Segment_ID and Default_Ind for a given date
    :param segment_id: name of the Segment_ID columns
    :return: a list of 0/1 (1: for test is passed i.e. segments are different)
    """
    segments = sorted(df[segment_id].unique())
    res_mtrx = [[np.nan for _ in  range(len(segments))] for _ in  range(len(segments))]
    res_mtrx_pvalue = [[np.nan for _ in  range(len(segments))] for _ in  range(len(segments))]
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            seg_1 = df[df[segment_id] == segments[i]][col_target]
            seg_2 = df[df[segment_id] == segments[j]][col_target]
            res_mtrx[i][j] = heterogeneity_test_1to1(seg_1, seg_2, alpha)[1]
            res_mtrx_pvalue[i][j] = heterogeneity_test_1to1(seg_1, seg_2, alpha)[0]
    res_mtrx = pd.DataFrame(res_mtrx, columns=segments, index=segments)
    res_mtrx_pvalue = pd.DataFrame(res_mtrx_pvalue, columns=segments, index=segments) 
    res = 'Pass' if res_mtrx.all().all() else 'Fail'
    return res, res_mtrx, res_mtrx_pvalue

#%%
def statistical_test_mrs_mdl(df_sp_mrs, col_target, col_pred, z_test=True, gini=True, hhi=True, hetero=True, psi=True,col_snap_date = 'SL_Date'):
    # functionality: 
    #   conduct statistical test for the segmentation model.
    # inputs:
    #   df_sp_mrs: the snapshot data with MRS bins assigned
    #   col_target: the column name of the targe variable
    #   col_pred: the column name of the prediction
    #   z_test: indicator whether do z-test
    #   gini: indicator whether calculate gini
    #   hhi: indicator whether calculate hhi 
    #   hetero: indicator whether do heterogeneity test
    #   psi: indicator whether calculate psi
    # outputs:
    #   summarized results

    print('statistical_test_mrs_mdl')
    def gen_ztest_result(df_tst, test):
        print('statistical_test_mrs_mdl -> gen_ztest_result')
        res = []
        for tst in test :
            df = df_tst if tst=='All' else df_tst.query(f"Covid_Period_Flag=='Y'") if tst == 'COVID' else df_tst.query(f"Split=='{tst}'")
            res.append([df.groupby(col_snap_date)[col_pred].mean().mean(), 
                        df.groupby(col_snap_date)[col_target].mean().mean(), 
                        ztest(df[col_pred], df[col_target], value=0, alternative='smaller')[1]])
        res = pd.DataFrame(res, columns=['Average Predicated PD', 'DR', 'Z-test One Tail p-value'], index=test)
        return res
    def gen_gini_result(df_tst, test):
        print('statistical_test_mrs_mdl -> gen_gini_result')
        res = []
        for tst in test:
            df = df_tst if tst=='All' else df_tst.query(f"Covid_Period_Flag=='Y'") if tst == 'COVID' else df_tst.query(f"Split=='{tst}'")
            res.append([univariate_gini(df, 'Pred_'+col_target, col_target), univariate_gini(df, col_pred, col_target)])
        res = pd.DataFrame(res, columns=[f'Gini ({"Pred_"+col_target})', f'Gini ({col_pred})'], index=test)
        return res
    def cal_hhi(df_tst, test):
        print('statistical_test_mrs_mdl -> cal_hhi')
        res = []
        for tst in test:
            df = df_tst if tst=='All' else df_tst.query(f"Covid_Period_Flag=='Y'") if tst == 'COVID'  else df_tst.query(f"Split=='{tst}'")
            res.append([sum((x*100)**2 for x in df['MRS_Bin'].value_counts(normalize=True, dropna=False).values)])
        res = pd.DataFrame(res, columns=['HHI'], index=test)
        return res
    def hetero_test(df_tst, col_target, test):
        print('statistical_test_mrs_mdl -> hetero_test')
        res, res_mtrx, res_mtrx_pvalue = [], {},{}
        for tst in test:
            df = df_tst if tst=='All' else df_tst.query(f"Covid_Period_Flag=='Y'") if tst == 'COVID' else df_tst.query(f"Split=='{tst}'")
            res_tmp = heterogeneity_test_multi_seg(df, col_target, 'MRS_Bin')
            res.append(res_tmp[0])
            res_mtrx[tst] = res_tmp[1]
            res_mtrx_pvalue[tst] = res_tmp[2]
        res = pd.DataFrame(res, columns=['Heterogenity Test'], index=test)
        return res, res_mtrx, res_mtrx_pvalue
    def psi_test(df_tst, test):
        print('statistical_test_mrs_mdl -> psi')
        res = []
        df_base = df_tst.loc[df_tst['Split']=='TRN', 'MRS_Bin'].value_counts(normalize=True, dropna=False).reset_index()
        for tst in test:
            df = df_tst if tst=='All' else df_tst.query(f"Covid_Period_Flag=='Y'") if tst == 'COVID' else df_tst.query(f"Split=='{tst}'")
            df = df['MRS_Bin'].value_counts(normalize=True, dropna=False).reset_index()
            res.append(psi_cal(df, df_base, 'index', 'MRS_Bin'))
        res = pd.DataFrame(res, columns=['PSI (vs TRN)'], index=test)
        return res
    
    test = [x for x in list(df_sp_mrs['Split'].unique()) if x==x]+['All','COVID']
    test = sorted(test, key=lambda x: {tst: i for i, tst in enumerate(['TRN', 'OOS', 'OOT', 'All', 'COVID'])}.get(x, np.inf))
    df_tst = df_sp_mrs.copy()
    res = []
    if z_test:
        res.append(gen_ztest_result(df_tst, test))
    if gini:
        res.append(gen_gini_result(df_tst, test))
    if hhi:
        res.append(cal_hhi(df_tst, test))
    if psi:
        res.append(psi_test(df_tst, test))
    if hetero:
        res_hetero, res_hetero_mtrx, res_hetero_mtrx_pvalue = hetero_test(df_tst, col_target, test)
        res.append(res_hetero)
        return {'Summary': pd.concat(res, axis=1), 'Heterogeneity Matrix': res_hetero_mtrx, 'Heterogeneity Matrix Pvalue': res_hetero_mtrx_pvalue}
    return {'Summary': pd.concat(res, axis=1)}


#%% 

def add_MRS_Bin(df_for_mdl = None, pd_edges = None, Col_Target = None, dlq_addon = True, df_dlq_early = None, df_dlq_late = None):
    # Add MRS bins

    df_for_mdl['MRS_Bin'] = pd.cut(df_for_mdl['Pred_'+Col_Target], pd_edges, include_lowest=True, right=True).cat.codes+1  #Add segmentation for current accounts

    if dlq_addon:
        print('***************Add MRS Bins/Segments for Early and Late Delinquent Accounts **********************************************')

        df_dlq_early['MRS_Bin'] = df_for_mdl['MRS_Bin'].max()+1

        if combine_earlY_and_late_dlq:
            df_dlq_late['MRS_Bin'] = df_for_mdl['MRS_Bin'].max()+1
        else:    
            df_dlq_late['MRS_Bin'] = df_for_mdl['MRS_Bin'].max()+2

        df_all = pd.concat([df_for_mdl, df_dlq_early, df_dlq_late], axis=0).reset_index(drop=True)
    else:
        df_all = df_for_mdl

    return df_all


def apply_pd_mdl(df_for_mdl,mdl_pd, Col_Target, dlq_addon = True, df_dlq_early = None, df_dlq_late = None):

    df_for_mdl = add_model_prediction(df_for_mdl, mdl_pd['Score']).copy()   #Predicted PD by scoring model, only for current accounts!

    df_all = add_MRS_Bin(df_for_mdl = df_for_mdl, pd_edges = mdl_pd['MRS']['PD_Edges'], Col_Target = Col_Target, dlq_addon = dlq_addon, df_dlq_early = df_dlq_early, df_dlq_late = df_dlq_late)

    df_all['Mapped_PD'] = df_all['MRS_Bin'].map(mdl_pd['MRS']['Mapped_PD'])  # Add mapped PD for each MRS Bin/Segment
   
    return df_all
#%%
def mrs_bt_stat_testing(df_sp_mrs = None, var_test_PD = 'Mapped_PD', fig_dir = None, file_out = None):
    # All kinds of statistical testing
    # df_sp_mrs.loc[(df_sp_mrs[col_snap_date]>=Covid_Affect_Start_Date) & (df_sp_mrs[col_snap_date]<=Default_IS_End_Date), 'Split'] = 'COVID'   #Covid time period

    df_sp_mrs.loc[(df_sp_mrs[col_snap_date]>=Covid_Affect_Start_Date) & (df_sp_mrs[col_snap_date]<=Covid_Affect_End_Date), 'Covid_Period_Flag'] = 'Y'   #Covid time period

    res_mrs_dist_all = plot_MRS_Bin_overall_dist(df_sp_mrs, Col_Target, col_snap_date= col_snap_date, Fig_Dir = fig_dir, var_PD = var_test_PD )   # Portfolio level segment distribution

    res_mrs_dist_mthly, res_mrs_count_mthly, res_mrs_default_count_mthly = plot_MRS_Bin_mthly_dist(df_sp_mrs, col_snap_date= col_snap_date, Fig_Dir= fig_dir)  #Monthly segment distribution
    
    if 'SL_Quarter' in df_sp_mrs.columns:
        res_mrs_dist_qtrly, res_mrs_count_qtrly, res_mrs_default_count_qtrly = plot_MRS_Bin_mthly_dist(df_sp_mrs, col_snap_date= 'SL_Quarter', Fig_Dir= fig_dir)

    res_mrs_bin_dr = plot_mrs_bin_dr_backtesting(df_sp_mrs, col_target = Col_Target, col_snap_date = col_snap_date, Fig_Dir = fig_dir)

    if 'SL_Year' in df_sp_mrs.columns:

        res_mrs_bin_dr_yr = plot_mrs_bin_dr_backtesting(df_sp_mrs, col_target = Col_Target, col_snap_date = 'SL_Year', Fig_Dir = fig_dir)
    
    if 'SL_Quarter' in df_sp_mrs.columns:

        res_mrs_bin_dr_qt = plot_mrs_bin_dr_backtesting(df_sp_mrs, col_target = Col_Target, col_snap_date = 'SL_Quarter', Fig_Dir = fig_dir)

    res_mrs_dr_bt_port = pd_backtesting_plot_ci_err_stat(df_sp_mrs, Col_Target, var_test_PD, 
                                                              alpha=0.05, test='z-test', tail=1, cols_seg=[], qrt_end=True, Fig_Dir = fig_dir)

    res_mrs_dr_bt_seg = pd_backtesting_plot_ci_err_stat(df_sp_mrs, Col_Target, var_test_PD, 
                                                              alpha=0.05, test='z-test', tail=1, cols_seg=['MRS_Bin'], qrt_end=True, Fig_Dir = fig_dir)

    res_statistical_test = statistical_test_mrs_mdl(df_sp_mrs= df_sp_mrs,col_target= Col_Target, col_pred= var_test_PD, z_test= True, gini=True, hhi= True, hetero= True, psi= True,col_snap_date = col_snap_date)


    res_psi = gen_psi_mthly(df = df_sp_mrs, col_target='MRS_Bin', window_base= [Data_Collection_Start_Date, Data_Collection_End_Date], Fig_file_out= fig_dir+'\\PSI_Monthly_MRS_Bin.png') #PSI is movement against the data df restricted to the period specified by window_base


    res_mrs_tst = {'Dist_All': res_mrs_dist_all, 'Dist_Mthly': res_mrs_dist_mthly, 
                            'MRS Bin Backtesting': res_mrs_bin_dr, 'Stat_Test': res_statistical_test,
                            'MRS DR Backtesting Port Lvl': res_mrs_dr_bt_port,
                            'MRS DR Backtesting Seg Lvl': res_mrs_dr_bt_seg
                            }

    # write results into Exel file
    with pd.ExcelWriter(file_out) as Writer:
        print('*************************Save Excel files for Back-testing ************************************')
        res_mrs_dist_all.to_excel(Writer, sheet_name = 'Dist_All')

        # write monthly distribution, data counts and default counts
        startcol = 0
        sheet_name = 'Dist_Mthly'

        pd.DataFrame(['Monthly Distribution'], columns=['Label']).to_excel(Writer, sheet_name = sheet_name, startcol = startcol, startrow = 0, index=False, header= False)

        res_mrs_dist_mthly.to_excel(Writer, sheet_name = sheet_name, startcol = startcol, startrow = 1)

        startcol = startcol + res_mrs_dist_mthly.shape[1]+2

        pd.DataFrame(['Monthly Data Count'], columns=['Label']).to_excel(Writer, sheet_name = sheet_name, startcol = startcol, startrow = 0, index=False, header= False)

        res_mrs_count_mthly.to_excel(Writer, sheet_name = sheet_name, startcol = startcol, startrow = 1)

        startcol = startcol + res_mrs_count_mthly.shape[1]+2

        pd.DataFrame(['Monthly Default Count'], columns=['Label']).to_excel(Writer, sheet_name = sheet_name, startcol = startcol, startrow = 0, index=False, header= False)

        res_mrs_default_count_mthly.to_excel(Writer, sheet_name = sheet_name, startcol = startcol, startrow = 1)

        # Bin Default Rate
        res_mrs_bin_dr.to_excel(Writer, sheet_name = 'Bin_DR')

        # statistical tests
        res_statistical_test['Summary'].to_excel(Writer, sheet_name = 'Perf_Test')
        if res_statistical_test.get('Heterogeneity Matrix') is not None:   # Dictionary requires special treatment before writing to Excel
            startcol = 1
            for key,val in res_statistical_test.get('Heterogeneity Matrix').items():  
                df = res_statistical_test['Heterogeneity Matrix'][key].copy()
                df['Data Set'] = key              
                df.rename_axis('MRS Bin',inplace= True)
                df.to_excel(Writer, sheet_name = 'Het_Test', startcol = startcol)  
                startcol = startcol +  df.shape[1] + 4 

            startcol = 1
            for key,val in res_statistical_test.get('Heterogeneity Matrix Pvalue').items():  
                df = res_statistical_test['Heterogeneity Matrix Pvalue'][key].copy()
                df['Data Set'] = key              
                df.rename_axis('MRS Bin',inplace= True)
                df.to_excel(Writer, sheet_name = 'Het_Test_Pvalue', startcol = startcol)  
                startcol = startcol +  df.shape[1] + 4   

        res_mrs_dr_bt_port['Summary'].to_excel(Writer, sheet_name = 'CITest_Port', index = False, startcol=1)
        startcol= res_mrs_dr_bt_port['Summary'].shape[1] + 4
        for key,val in res_mrs_dr_bt_port['Detail'].items(): # Dictionary requires special treatment before writing to Excel
            df = res_mrs_dr_bt_port['Detail'][key].copy()
            df['Data Set'] = key 
            df.to_excel(Writer, sheet_name = 'BKTest_Port', index = False, startcol = startcol)  
            startcol = startcol + df.shape[1] + 2   

        res_mrs_dr_bt_seg['Summary'].to_excel(Writer, sheet_name = 'CITest_Seg', index = False, startcol = 1)
        startcol= res_mrs_dr_bt_seg['Summary'].shape[1] + 4
        for key,val in res_mrs_dr_bt_seg['Detail'].items(): # Dictionary requires special treatment before writing to Excel
            df = res_mrs_dr_bt_seg['Detail'][key].copy()
            df['Data Set'] = key 
            df.to_excel(Writer, sheet_name = 'BKTest_Seg', index = False, startcol = startcol)  
            startcol = startcol + df.shape[1] + 2   
    return res_statistical_test    
    
####################################################################################################################################################

###############################Sentitivity and Stress Testing Functions ###########################################################################

#########################################################################################################################################################

#%%
def mi_cal(ss_from, ss_to):
    # functionality: 
    #   calculate the mobility index.
    # inputs:
    #   ss_from: the series of the distribution before the migration
    #   ss_from: the series of the distribution after the migration
    # outputs:
    #   mi: the mobility index
    print('mi_cal')
    ss_from.reset_index(drop=True, inplace=True)
    ss_to.reset_index(drop=True, inplace=True)
    cross_tab = pd.crosstab(ss_from, ss_to, normalize='index')
    rank_from = cross_tab.index   # the migration columns has to be numeric, or coverted to numeric before doing migration calculation
    rank_to = cross_tab.columns
    trans_mtrx = cross_tab.values
    k= len(trans_mtrx)
    ni_n = ss_from.value_counts(normalize=True).values
    mi = 1/(k-1) * np.sqrt(sum(ni_n[i] * trans_mtrx[i, j] * (rank_from[i]-rank_to[j])**2 for i in range(k) for j in range(trans_mtrx.shape[1])))
    return mi

#%%
def psi_vs_base(df_shock, df_base, col_segment):
    # functionality: 
    #   get the segment psi for dataframes before and after shock 
    # inputs:
    #   df_shock: after shock dataframe
    #   df_base: before shock dataframe
    #   col_segment: the column name for the sengemts 
    # outputs:
    #   psi

    print('psi_vs_base')
    df_actl = pd.DataFrame(df_shock[col_segment].value_counts(normalize=True)).reset_index()
    df_expect = pd.DataFrame(df_base[col_segment].value_counts(normalize=True)).reset_index()
    return psi_cal(df_actl, df_expect, 'index', col_segment)

#%%
def shock_analysis(df_base, df_shock, pct, col_pred, col_segment):

    df_base_new = df_base.rename(columns = {col_segment: col_segment + '_base'})
    df_shock_new = df_shock.rename(columns = {col_segment: col_segment + '_shock', col_pred: col_pred+'_shock'})

    df_join = df_base_new[[Col_Loan_Nbr, col_snap_date,col_segment + '_base']].merge(right = df_shock_new[[Col_Loan_Nbr, col_snap_date,col_segment + '_shock', col_pred + '_shock']], on = [Col_Loan_Nbr, col_snap_date], how = 'left')

    df_base = df_join[[Col_Loan_Nbr, col_snap_date,col_segment + '_base']].rename(columns = {col_segment + '_base': col_segment})
    df_shock = df_join[[Col_Loan_Nbr, col_snap_date,col_segment + '_shock',col_pred+'_shock']].rename(columns = {col_segment + '_shock': col_segment, col_pred+'_shock': col_pred})

    return {f'Average Mapped PD, {pct*100}%': df_shock[col_pred].mean(),
            f'PSI, {pct*100}%': psi_vs_base(df_shock, df_base, col_segment),
            f'MI, {pct*100}%': mi_cal(df_base[col_segment], df_shock[col_segment])}

#%%
def migrate_pct_woe_bin_sensi_stress(df, col_woe, pct, woe_mapping):
    # functionality: 
    #   migrate x percent of the population of each woe bin to the adjacent bin
    # inputs:
    #   df: snapshot data with woe
    #   pct: the percentage of the migration
    #   woe_mapping: the woe mapping of the driver
    # outputs:
    #   df: the after migration dataframe.
    df.reset_index(drop=True, inplace=True)
    for i, woe in woe_mapping.items():
        if not((pct<0 and i==0) or (pct>0 and i==len(woe_mapping)-1)):
            df_bin = df.loc[round(df[col_woe], 6)==round(woe_mapping[i],6)]
            df_sample = df_bin.sample(abs(int(len(df_bin)*pct)), random_state=1)
            df.loc[df.index.isin(df_sample.index), col_woe] = woe_mapping[i+math.copysign(1, pct)]
    return df

#%%
def get_mdl_num_cols(mdl, df):
    # functionality: 
    #   get the model's numeric drivers
    # inputs:
    #   mdl: the model
    #   df: the dataframe fits for the model
    # outputs:
    #   list of the numeric drivers

    print('get_mdl_num_cols')
    cols_driver = [x[:-4] for x in mdl.model.exog_names if x!='const' and x[-4:]=='_WOE']
    cols_num = set()
    for col in cols_driver:
        if col in df.columns and df[col].dtype in ('int64', 'float64'):
            cols_num.add(col)
    return list(cols_num)


#%%
def shock_all_woe_cols(df_in, df_base, mdl_pd, col_target, cols_woe, col_pred, col_segment, codebook, pct_range, dlq_addon = True, df_dlq_early = None, df_dlq_late = None ):
    # functionality: 
    #   shock all woe drivers for stress testing
    # inputs:
    #   df: snapshot data with woe
    #   pct: the percentage of the migration
    #   woe_mapping: the woe mapping of the driver
    # outputs:
    #   df: the after migration dataframe.
    res = {'Variable': 'Stress Testing'}
    for pct in pct_range:
        print(f'sensitivity_analysis_num_driver ->shock_all_woe_cols: {pct*100}%')
        if pct!=0:
            df_tst = copy.deepcopy(df_in)
            if codebook is not None:
                for col_woe in cols_woe:
                    df_tst = migrate_pct_woe_bin_sensi_stress(df_tst, col_woe, pct, codebook[col_woe[:-4]])
            df_mrs_pred = apply_pd_mdl(df_tst, mdl_pd, col_target, dlq_addon = dlq_addon, df_dlq_early = df_dlq_early, df_dlq_late = df_dlq_late)
        else:
            df_mrs_pred = df_base
        res.update(shock_analysis(df_base, df_mrs_pred, pct, col_pred, col_segment))
    res = pd.DataFrame(res, index=[0])
    return res

#%%
def get_woe_codebook_from_data(df):
    # functionality: 
    #   get the woe codebook from data 
    # inputs:
    #   df: the dataframe contains the woe
    # outputs:
    #   dict_codebook: the dictionary version of the codebook

    codebook = []
    for col in df.columns:
        if col[-4:]=='_WOE':
            df_woe = df[[col]].drop_duplicates()
            df_woe['Variable_Name'] = col[:-4]
            df_woe.rename(columns={col: 'WOE'}, inplace=True)
            codebook.append(df_woe)
    if not codebook:
        return None
    codebook = pd.concat(codebook, axis=0)
    codebook.sort_values(['Variable_Name', 'WOE'], inplace=True)
    codebook['Bin'] = codebook.groupby('Variable_Name').cumcount()
    dict_codebook = {}
    for var in codebook['Variable_Name'].values:
        bin_code = codebook.loc[codebook['Variable_Name']==var, ['Bin', 'WOE']].drop_duplicates()
        dict_codebook[var] = {int(i): woe for i, woe in bin_code.values}
    return dict_codebook
        
def data_trim(df_in, df_base, cols_driver, exclude_covid, dlq_addon= False, df_dlq_early = None, df_dlq_late = None):
    print('data_trim')
    if exclude_covid:
        query_code = 'not("'+f"{Covid_Affect_Start_Date}"+'"<='+f"{col_snap_date}"+'<="'+f"{Default_IS_End_Date}"+'")'   # Exclude Covid only from In-Sample if needed because OOT can allow Covid included
        df = df_in.query(query_code).copy()
        df_base = df_base.query(query_code).copy() 
        if dlq_addon:
            df_dlq_early = df_dlq_early.query(query_code).copy()
            df_dlq_late = df_dlq_late.query(query_code).copy()
    else:
        df = df_in
        df_dlq_early = df_dlq_early
        df_dlq_late = df_dlq_late
        df_base = df_base

    df_in_trim = df[df.columns[df.columns.isin(cols_driver)]]
    return df_in_trim, df_base, df_dlq_early, df_dlq_late
    
def shock_1_num_col(PCT_RANGE, df_for_mdl, df_base, mdl_pd, col_target, col_num, col_pred, col_segment, file_num_woe_codebook, file_cat_woe_codebook, dlq_addon = False, df_dlq_early = None, df_dlq_late = None):
    res = {'Variable': col_num}
    for pct in PCT_RANGE:
        print(f'sensitivity_analysis_num_driver -> shock_1_num_dr: {col_num} {pct*100}%')
        if pct!=0:
            df_tst = copy.deepcopy(df_for_mdl)
            if col_num+'_WOE' in mdl_pd['Score'].model.exog_names:
                df_tst[col_num] *= (1+pct)
                df_woe_tmp = add_woe(df = df_tst[[col_num]], fileIn_Num= file_num_woe_codebook, fileIn_Cat=file_cat_woe_codebook)

                df_tst[col_num+'_WOE'] = df_woe_tmp[col_num+'_WOE']
                
            df_mrs_pred = apply_pd_mdl(df_tst, mdl_pd, col_target, dlq_addon=dlq_addon, df_dlq_early= df_dlq_early, df_dlq_late=df_dlq_late)
        else:
            df_mrs_pred = df_base

        
        res.update(shock_analysis(df_base, df_mrs_pred, pct, col_pred, col_segment))
    res = pd.DataFrame(res, index=[0])
    return res  

def get_mdl_woe_cols(mdl_score):
    print('get_mdl_woe_cols')
    return [x for x in mdl_score.model.exog_names if x!='const' and x[-4:]=='_WOE']

def shock_1_woe_col(PCT_RANGE,df_for_mdl, df_base, mdl_pd, col_target, col_woe, col_pred, col_segment, codebook, dlq_addon = False, df_dlq_early = None, df_dlq_late = None):
    res = {'Variable': col_woe}
    for pct in PCT_RANGE:
        print(f'sensitivity_analysis_num_driver -> shock_1_woe_col: {col_woe} {pct*100}%')
        if pct!=0:
            df_tst = copy.deepcopy(df_for_mdl)
            if codebook is not None:
                df_tst = migrate_pct_woe_bin_sensi_stress(df_tst, col_woe, pct, codebook[col_woe[:-4]])
            if dlq_addon:
                df_mrs_pred = apply_pd_mdl(df_tst, mdl_pd, col_target,dlq_addon = dlq_addon, df_dlq_early = df_dlq_early, df_dlq_late = df_dlq_late)
        else:
            df_mrs_pred = df_base
        res.update(shock_analysis(df_base, df_mrs_pred, pct, col_pred, col_segment))
    res = pd.DataFrame(res, index=[0])
    return res


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#  Function for Impact Analysis, due to outlier or other data checking

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def add_outlier_ind(df, col, test='Outer_Fence'):
    # functionality: 
    #   add the outlier indicator to the dataframe
    # inputs: 
    #   df: the input dataframe
    #   col: the tested column name 
    #   test: indicator the outlier type
    # outputs:
    #   df: the dataframe with the outlier indicator added
    
    df = df.reset_index(drop=True)
    if len(df[df[col]==0])/len(df)>0.5:
        # for arrear days and delinquency status, most will be 0 and shoul not be considered outlier
        df_tst = df[df[col]!=0]
    else:
        df_tst = df[[col]]
    q1 = df_tst[col].quantile(0.25)
    q2 = df_tst[col].quantile(0.5)
    q3 = df_tst[col].quantile(0.75)
    if test in ('Outer_Fence', 'Inner_Fence'):
        mul = 3 if test=='Outer_Fence' else 1.5
        q_lb, q_ub = q1 - mul*(q3-q1), q3 + mul*(q3-q1)
        df_tst[f'{col}_Outlier_Ind'] = df_tst[col].apply(lambda x: x < q_lb or x > q_ub)
    elif test=='X84':
        mad = abs(df_tst[col]-q2).median()
        df_tst[f'{col}_Outlier_Ind'] = df_tst[col].apply(lambda x: (x-q2)>5.2*mad) 
    else:
        raise ValueError(f'Wrong test={test}! test should be in ("Original", "Outer_Fence", "Inner_Fence", "X84")')
    df = df.merge(df_tst[[f'{col}_Outlier_Ind']], left_index=True, right_index=True, how='left')
    return df    


def get_mdl_drivers(mdl):
    cols_driver = [x[:-4] for x in mdl.model.exog_names if x!='const' and x[-4:]=='_WOE'] 
    return cols_driver      

def remove_outlier(df, mdl, outlier):
    cols_num = get_mdl_num_cols(mdl, df)
    for col in cols_num:
        df = add_outlier_ind(df, col, test=outlier)
    df = df[~df[[f'{col}_Outlier_Ind' for col in cols_num]].any(axis=1)]
    df.drop(columns=[f'{col}_Outlier_Ind' for col in cols_num], inplace=True)
    return df

def impact_analysis(pd_mdl_old, pd_mdl_new, df_pred, df_base, exclude_covid, file_out = None):
    if exclude_covid:
        query_code = 'not("'+f"{Covid_Affect_Start_Date}"+'"<='+f"{col_snap_date}"+'<="'+f"{Covid_Affect_End_Date}"+'")'  #whether to conduct impact analysis on data including Covid or not
        df_pred = df_pred.query(query_code) 
        df_base = df_base.query(query_code)  
    pd_seg_new = df_pred.groupby('MRS_Bin', as_index=False)[['Mapped_PD']].agg(('count', 'mean'))
    pd_seg_base = df_base.groupby('MRS_Bin', as_index=False)[['Mapped_PD']].agg(('count', 'mean'))
    pd_seg_cmp = pd_seg_new.merge(pd_seg_base, on='MRS_Bin', suffixes=['_New', '_Original'], how = 'outer')     
    mdl_old = pd_mdl_old['Stat'][['Feature', 'Coefficients', 'Relative Weight']]
    mdl_new = pd_mdl_new['Stat'][['Feature', 'Coefficients', 'Relative Weight']]
    stat_cmp = mdl_new.merge(mdl_old, on='Feature', suffixes=['_New', '_Original'],how = 'outer')
    stat_cmp['Coefficient_Change'] = stat_cmp['Coefficients_New']-stat_cmp['Coefficients_Original']
    stat_cmp['Coefficient_Change%'] = stat_cmp['Coefficient_Change']/stat_cmp['Coefficients_Original']
    stat_cmp['Relative_Weight_Change'] = stat_cmp['Relative Weight_New']-stat_cmp['Relative Weight_Original']
    stat_cmp['Relative_Weight_Change%'] = stat_cmp['Relative_Weight_Change']/stat_cmp['Relative Weight_Original']
    stat_cmp['Relative_Weight_PSI'] = psi_cal(mdl_old, mdl_new, 'Feature', 'Relative Weight')
    stat_cmp['Segment_Bin_MI'] = mi_cal(df_pred['MRS_Bin'].copy(), df_base['MRS_Bin'].copy())
    stat_cmp['Average_PD_New'] = df_pred['Mapped_PD'].mean()
    stat_cmp['Average_PD_Original'] = df_base['Mapped_PD'].mean()

    if file_out is not None:

        with pd.ExcelWriter(file_out) as Writer:
            print(f'*************************Save Excel files for impact Analysis************************************')

            pd_seg_cmp.to_excel(Writer, sheet_name = 'Seg_Cmp')
            stat_cmp.to_excel(Writer, sheet_name = 'Model_Cmp')

    return {'PD_Seg': pd_seg_cmp, 'Model_Cmp': stat_cmp}       


def woe_binning_automatic(df = pd.DataFrame(),file_woe = {}):
    # WOE binning for df
    # Note: the following codes are based on the univariate binning in S6_Univariate_Analysis. Must be consistent to ensure the same binning was applied
    woe_num_GiniBinning = gen_woe_binning_numeric_drivers(df,[], Col_Target, cols_skip, n_max_bins = max_woe_bins, n_ini_cuts=100, min_bin_pd=0.0003, alpha=0.15, min_bin_prob=min_woe_bin_count/df.shape[0], min_woe_diff= min_woe_diff)

    woe_num_GiniBinning.to_csv(file_woe['num'],index = False)

    woe_cat_NoBinning = gen_woe_categorical_drivers(df, [], Col_Target, cols_skip)  

    woe_cat_NoBinning.to_csv(file_woe['cat_ini'])      

    cols_nonBinned = ['Combo_Province_Metro_Override'] #the grouping is based on business feedback (20230831) and therefore not going through further binning by the algorithm

    woe_cat_WoeDiffBinning = binning_cat_woe_by_woeDiff(file_woe['cat_ini'],Col_Target, 
                                                min_woe_diff = min_woe_diff, min_bin_prob = min_woe_bin_count/df.shape[0], 
                                                file_out = file_woe['cat'],
                                                cols_skip = cols_nonBinned)

    
    # WOE binning Adjustment

    codebook_num = pd.read_csv(file_woe['num'], low_memory=False) 
    
    try: 
        codebook_cat = pd.read_csv(file_woe['cat'], low_memory=False) 
    except pd.errors.EmptyDataError:
        print(f'******************{file_woe["cat"]}'+' is empty********************')
        codebook_cat = pd.DataFrame()

    #BNI variables WOE will take data driven WOE unless Beacon is missing as well. This treatment makes sense because BNI is not typically required from borrower unless Beacon is missing. This treatment is taken care in override_WOE in add WOE step
    col_skip_woe_adj_num = list(set(codebook_num['Variable_Name'][codebook_num['Variable_Name'].str.contains('BNI')]))
    print('*Penalize Missing Bins by assigning worst WOE among all auto calibrated woes*')

    codebook_num_adj = penalize_missing_bin(codebook_num, col_bin='Bin', col_woe='Bin_WOE', val_missing=-1, col_skip= col_skip_woe_adj_num)

    if not codebook_cat.empty:
        codebook_cat_adj = penalize_missing_bin(codebook_cat, col_bin='Value', col_woe='WOE', val_missing=np.nan)
    else:
        codebook_cat_adj = pd.DataFrame()

    print('******WOE override for LoanPurpose_Override ************')
    # Assign neutral WOE (=0) for missing and OT (other) for LoanPurpose because they are likely due to third party loans (expected to be missing) and also their default rates  are in the lower ranges compared to other bins in the preliminary analysis

    if not codebook_cat_adj.empty:
        cond_mis_or_OT = (codebook_cat_adj['Variable_Name']=='LoanPurpose_Override') & (codebook_cat_adj['Value'].isna())

        codebook_cat_adj.loc[cond_mis_or_OT, 'WOE'] = 0

    print('*save adjuted woe*')

    codebook_num_adj.to_csv(file_woe['num_adj'], index=False)

    codebook_cat_adj.to_csv(file_woe['cat_adj'], index=False)

    # Final binning without manual binning override (because manual binning requires business review - which is not available during automatic scenario analysis)

    print('*Final WOE the same as adjusted woe*')

    codebook_num_adj.to_csv(file_woe['num_final'], index=False)

    codebook_cat_adj.to_csv(file_woe['cat_final'], index=False)


    df = add_woe(df, file_woe['num_final'], file_woe['cat_final'], file_out=None, penalize_new = True) # if the value did not exist in training, then penalize with the worst WOE value

    return df


def MRS_Model_training(df_mrs_trn_w_score = None, Col_Target = None, max_mrs_bins = None, min_mrs_bin_dist = None, dlq_addon = None, df_mrs_trn_dlq_early = None, df_mrs_trn_dlq_late = None, col_snap_date = None,edges = None):
    # Develop MRS Model

    edges_new = edges
        
    df_all_mrs_train = add_MRS_Bin(df_for_mdl = df_mrs_trn_w_score, pd_edges = edges_new, Col_Target = Col_Target, dlq_addon = dlq_addon, df_dlq_early = df_mrs_trn_dlq_early, df_dlq_late = df_mrs_trn_dlq_late)

    mapped_PD_new,PD_mrs_agg_new = cal_MRS_Bin_mapped_pd(df_all_mrs_train, col_dft_ind=Col_Target, col_date = col_snap_date)

    MRS_summary_new = [[i, edges_new[i-1], edges_new[i], mapped_PD_new.get(i,np.nan), PD_mrs_agg_new.get(i, np.nan)] for i in range(1, len(edges_new))]

    if dlq_addon:
        # No lower and upper bound for mapped PD for early and late delinquent accounts
        if combine_earlY_and_late_dlq:
            MRS_summary_new.append([max(mapped_PD_new.keys()), 'Delinquent Accounts', 'Delinquent Accounts', mapped_PD_new[max(mapped_PD_new.keys())], PD_mrs_agg_new[max(mapped_PD_new.keys())]])

        else:    
            MRS_summary_new.append([max(mapped_PD_new.keys())-1, 'Early Delinquent Account', 'Early Delinquent Account', mapped_PD_new[max(mapped_PD_new.keys())-1], PD_mrs_agg_new[max(mapped_PD_new.keys())-1]])

            MRS_summary_new.append([max(mapped_PD_new.keys()), 'Late Delinquent Account', 'Late Delinquent Account', mapped_PD_new[max(mapped_PD_new.keys())], PD_mrs_agg_new[max(mapped_PD_new.keys())]])

    MRS_summary_new = pd.DataFrame(MRS_summary_new, columns=["MRS_Bin", 'Lower_Bound', 'Upper_Bound', 'Mapped PD','MRS Aggregated PD'])

    res= {'PD_Edges': edges_new, 'Mapped_PD': mapped_PD_new, 'Summary': MRS_summary_new}

    return res

def model_result_w_new_data(df_scoring_trn_new= None, cols_driver = None, file_woe = None, df_mrs_dlq_cur=None, dlq_addon = False, df_mrs_dlq_early = None, df_mrs_dlq_late = None, pd_edges = None):
    # new model (both scoring and MRS) with new data
    # scoring model based on cols_driver

    print('**********************WOE Binning**************************************')
    df_scoring_trn_new = woe_binning_automatic(df = df_scoring_trn_new, file_woe = file_woe)

    pd_mdl_new = collections.defaultdict(dict)

    cols_woe = [x+'_WOE' for x in cols_driver]
    print('**************Training Scoring Model***************************')
    scoring_mdl_new = model_training(df_scoring_trn_new, Col_Target, cols_woe)
    
    pd_mdl_new['Score'] = scoring_mdl_new['Model']
    pd_mdl_new['Stat']  = scoring_mdl_new['Stat']

    print('***************Scoring MRS data with new model *******************')
    df_mrs_dlq_cur = add_woe(df_mrs_dlq_cur, file_woe['num_final'], file_woe['cat_final'], file_out=None, penalize_new = True) # if the value did not exist in training, then penalize with the worst WOE value
    df_mrs_trn_dlq_cur = df_mrs_dlq_cur.query('Split=="TRN"')

    df_mrs_trn_dlq_cur = add_model_prediction(df_mrs_trn_dlq_cur, scoring_mdl_new['Model'])

    print('*************Preparing Delinquent Training Data If needed *************************')

    if dlq_addon:

        df_mrs_trn_dlq_early = df_mrs_dlq_early.query('Split=="TRN"')

        df_mrs_trn_dlq_late = df_mrs_dlq_late.query('Split=="TRN"')

    print('**********Develop Segmentation model *********************')
    pd_mdl_new['MRS'] = MRS_Model_training(df_mrs_trn_w_score = df_mrs_trn_dlq_cur, Col_Target = Col_Target, max_mrs_bins = max_mrs_bins, min_mrs_bin_dist = min_mrs_bin_dist, dlq_addon = dlq_addon, df_mrs_trn_dlq_early = df_mrs_trn_dlq_early, df_mrs_trn_dlq_late = df_mrs_trn_dlq_late, col_snap_date = col_snap_date, edges= pd_edges)

    print('*********************Appply New Scoring and Segmentation Model******************************')
    df_pred = apply_pd_mdl(df_mrs_dlq_cur, pd_mdl_new, Col_Target, dlq_addon, df_mrs_dlq_early, df_mrs_dlq_late)  #apply scoring and segmentation model

    return pd_mdl_new, df_pred


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#  Function for Margin of Conservatism and Parameter Calibration

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%
def moc_default_rate_simulation(ss, n_sim=9999, quantile=0.6):
    # bootstrap samples and calculate mean of bootstrapped samples
    # then calculate 
    print('moc_default_rate_simulation')
    resi = stats.bootstrap((ss.values,), np.mean, n_resamples=int(n_sim), random_state=1)
    moc_res = pd.Series(resi.bootstrap_distribution).quantile(quantile)/ss.mean()-1
    return {'MOC_Residual': moc_res, 'Bootstraping': resi, 'Simulation': pd.Series(resi.bootstrap_distribution).quantile(quantile), 'Target': ss.mean(), 'Simulation 70% Qunatile': pd.Series(resi.bootstrap_distribution).quantile(0.7)} 


def moc_seg_lvl_default_rate_simulation(df, col_seg, col_dr, n_sim=9999, quantile=0.6, file_out = None):
    print('moc_seg_lvl_default_rate_simulation')
    res = []
    for seg in sorted(df[col_seg].unique()):
        resi = moc_default_rate_simulation(df.loc[df[col_seg]==seg, col_dr], n_sim, quantile)
        res.append([seg, resi['MOC_Residual'], resi['Simulation'], resi['Target'], resi['Simulation 70% Qunatile']])
    res = pd.DataFrame(res, columns=[col_seg, 'MOC', 'Sim', 'Target', 'Sim 70% Quantile'])
    if file_out is not None:
        res[[col_seg, 'MOC']].to_csv(file_out, index=False)
    return {'MOC_Residual': res}


#%%
def cal_total_moc(dict_moc, col_seg='', file_out = None):
    def process_moc_res_adj(dict_moc):
        # except for MOC_residual and adj_Longrun which are taken with separate name
        # the rest will be either added to seg_moc or port_moc, depending on data types (dataframe or not),
        # returning tot_moc, structure consistent with seg_moc
        seg_moc = None
        port_moc = 0
        for key, val in dict_moc.items():
            if key=='MOC_Residual':
                moc_resi = dict_moc['MOC_Residual']
                continue
            if key=='Adj_Longrun':
                adj_longrun = dict_moc['Adj_Longrun']
                continue
            if type(val)==type(pd.DataFrame()):
                if seg_moc is None:
                    seg_moc = val.copy()
                else:
                    seg_moc = seg_moc.merge(val, on='MRS_Bin', how='left', suffixes=['', '_add'])
                    seg_moc['MOC'] = seg_moc[['MOC', 'MOC_add']].apply(sum, axis=1)
                    seg_moc.drop(columns=['MOC_add'], inplace=True)
            else:
                port_moc += val
        return seg_moc, port_moc, moc_resi, adj_longrun
    def cal_total_moc_pre_adj(seg_moc, port_moc, moc_resi):
        if seg_moc is not None:
            # add port_moc to seg_moc and also add moc_resi to seg_moc by root squared sum 
            seg_moc['MOC'] += port_moc
            if type(moc_resi)==type(pd.DataFrame()): 
                seg_moc = seg_moc.merge(moc_resi, on='MRS_Bin', how='left', suffixes=['', '_Resi'])
                seg_moc['MOC'] = seg_moc[['MOC', 'MOC_Resi']].apply(lambda x: np.sqrt(x[0]**2+x[1]**2), axis=1)
                seg_moc.drop(columns=['MOC_Resi'], inplace=True)
            else:
                seg_moc['MOC'] = seg_moc['MOC'].apply(lambda x: np.sqrt(x**2+moc_resi**2))
            total_moc = seg_moc
        else:
            # add port_moc to moc_resi, and structure consistent with moc_resi (either scalar or data frame),
            # returning tot_moc
            if type(moc_resi)==type(pd.DataFrame()): 
                moc_resi['MOC'] = moc_resi['MOC'].apply(lambda x: np.sqrt(x**2+port_moc**2))
                total_moc = moc_resi
            else:
                total_moc = np.sqrt(port_moc**2+moc_resi**2)
        return total_moc
    def cal_total_moc_post_adj(total_moc, adj_longrun):
        if type(total_moc)==type(pd.DataFrame()) and type(adj_longrun)==type(pd.DataFrame()):
            # add longrun adjustment to total_moc
            total_moc_adj = total_moc.merge(adj_longrun, on='MRS_Bin', how='left')
            total_moc_adj['MOC'] = total_moc_adj[['MOC', 'Adj_Longrun']].apply(sum, axis=1)
            total_moc_adj.drop(columns=['Adj_Longrun'], inplace=True)
        elif type(adj_longrun)==type(pd.DataFrame()):
            total_moc_adj = adj_longrun.copy()
            total_moc_adj.rename(columns={'Adj_Longrun': 'MOC'}, inplace=True)
            total_moc_adj['MOC'] += total_moc
        elif type(total_moc)==type(pd.DataFrame()):
            total_moc_adj = total_moc.copy()
            total_moc_adj['MOC'] += adj_longrun
        else:
            total_moc_adj = total_moc+adj_longrun
        return total_moc_adj
    
    seg_moc, port_moc, moc_resi, adj_longrun = process_moc_res_adj(dict_moc)
    total_moc = cal_total_moc_pre_adj(seg_moc, port_moc, moc_resi) #total_moc before longrun adjustmnet
    total_moc = total_moc[['MRS_Bin','MOC']]
    # total_moc['MOC_Pre_Longrun_Adj']
    total_moc_adj = cal_total_moc_post_adj(total_moc, adj_longrun) # add long run adjustment and total_moc
    dict_moc.update({'MOC_Total': total_moc_adj, 'MOC_Total_Pre_Adj': total_moc})  
    if file_out is not None:
        with open(file_out, 'wb') as file:
            pickle.dump(dict_moc, file)
    return dict_moc  

#%%
def apply_moc_2_pd(df, file_moc, file_out=None, col_pd='Mapped_PD', Heterogeneity_Adj_Moc=False):
    with open(file_moc, 'rb') as file:
        moc_dict = pickle.load(file)
    moc = moc_dict['MOC_Total']
    moc_pre_Adj = moc_dict['MOC_Total_Pre_Adj']

    df.rename(columns={col_pd: 'PD_Pre_MOC'}, inplace=True)

    # Adding PD Pre Longrun Adjustment (post POC) 
    if type(moc_pre_Adj) in (int, float):
        df['PD_Post_MOC_Pre_Adj'] = df['PD_Pre_MOC'] * (1+ (0 if pd.isna(moc_pre_Adj) else moc_pre_Adj))
    else:
        df = df.merge(moc_pre_Adj, on='MRS_Bin', how='left').rename(columns = {'MOC':'MOC_Pre_Adj'})
        df['PD_Post_MOC_Pre_Adj'] = df[['PD_Pre_MOC', 'MOC_Pre_Adj']].apply(lambda x: x[0] * (1+( 0 if pd.isna(x[1]) else x[1])), axis=1) 

    # Adding PD Post MOC (and Post long-run adjustmnent)
    if type(moc) in (int, float):
        df['PD_Post_MOC'] = df['PD_Pre_MOC'] * (1+ (0 if pd.isna(moc) else moc))
    else:
        df = df.merge(moc, on='MRS_Bin', how='left')
        df['PD_Post_MOC'] = df[['PD_Pre_MOC', 'MOC']].apply(lambda x: x[0] * (1+( 0 if pd.isna(x[1]) else x[1])), axis=1) 
    
    if Heterogeneity_Adj_Moc:
        df['Absol_increment'] =  df['PD_Post_MOC'] - df['PD_Pre_MOC']
        MOC_monotonic = []
        for i in range(1,len(df['MRS_Bin'].unique())+1):
            temp = df.loc[df['MRS_Bin']<=i]['Absol_increment'].max()
            MOC_monotonic.append(temp)
        
        df['Absol_increment_mono'] = MOC_monotonic
        df['PD_Post_MOC'] = df['PD_Pre_MOC'] + df['Absol_increment_mono'] 
        
        df.drop(['Absol_increment_mono','Absol_increment'],axis =1,inplace=True)
        
    if file_out is not None:
        df.to_csv(file_out, index=False)
    return df

#%%
def apply_moc_2_pd_20230927(df, file_moc, file_out=None, col_pd='Mapped_PD'):
    with open(file_moc, 'rb') as file:
        moc = pickle.load(file)['MOC_Total']
    df.rename(columns={col_pd: 'PD_Pre_MOC'}, inplace=True)
    if type(moc) in (int, float):
        df['PD_Post_MOC'] = df['PD_Pre_MOC'] * (1+ (0 if pd.isna(moc) else moc))
    else:
        df = df.merge(moc, on='MRS_Bin', how='left')
        df['PD_Post_MOC'] = df[['PD_Pre_MOC', 'MOC']].apply(lambda x: x[0] * (1+( 0 if pd.isna(x[1]) else x[1])), axis=1) 
    if file_out is not None:
        df.to_csv(file_out, index=False)
    return df

#%%
def gen_psi_mthly(df, col_target, window_base, Fig_file_out = None):
    # PSI movement is against the period specified by window_base
    df_base = df.query(f"'{window_base[0]}' <= {col_snap_date} <= '{window_base[1]}'")
    df_base = pd.DataFrame(df_base[col_target].value_counts(normalize=True))
    df_base.reset_index(inplace=True)
    res = []
    for dt_sp in sorted(df[col_snap_date].unique()):
        df_mth = df[df[col_snap_date]==dt_sp]
        df_mth = pd.DataFrame(df_mth[col_target].value_counts(normalize=True))
        df_mth.reset_index(inplace=True)
        res.append([dt_sp, psi_cal(df_mth, df_base, 'index', col_target)])
    res = pd.DataFrame(res, columns=['Date', 'PSI'])
    res['Date'] = res['Date'].apply(pd.to_datetime)
    res['Migration'] = res['PSI'].apply(lambda x: 'Low' if x<=0.1 else ('Medium' if 0.1<x<=0.25 else 'High'))
    plt.plot(res['Date'], res['PSI'])
    plt.plot(res['Date'], [0.1]*len(res), linestyle=':', color='orange')
    plt.plot(res['Date'], [0.25]*len(res), linestyle=':', color='red')
    plt.legend(['PSI','Amber', 'Red'])
    plt.xlabel('Date')
    plt.ylabel('PSI')
    if Fig_file_out is not None:
        plt.savefig(Fig_file_out)
    # plt.ylim(top=1)
    # plt.show()
    plt.close()
    return res
#%%
def MOC_for_impact_analysis(impact_file_in = None):

    # caclculate MOC for impact analysis. The impact analysis file has a prescribed format, produced by the function tool.

    Moc_impact_01 = pd.read_excel(impact_file_in,sheet_name= 'Seg_Cmp')

    Moc_impact_02 = Moc_impact_01.drop(columns=['Mapped_PD_New','Mapped_PD_Original']).rename(columns = {'Unnamed: 0':'MRS_Bin','Unnamed: 2':'Mapped_PD_New', 'Unnamed: 4':'Mapped_PD_Original'})

    Moc_impact_03 = Moc_impact_02.iloc[2:,:]
    
    Moc_impact_04 = Moc_impact_03.sort_values(by= ['MRS_Bin'])

    Moc_impact_04['MOC'] = Moc_impact_04.loc[:,['Mapped_PD_New', 'Mapped_PD_Original']].apply(lambda x: max(x[0]/x[1] - 1,0), axis =1)

    Moc_impact = Moc_impact_04[['MRS_Bin','MOC']].replace(np.nan,0)

    return Moc_impact
    
#%%
def calibrate_param_pd(df_trn, col_seg, col_dt, col_dft, file_moc_total_pkl, file_mdl_pkl, seg_dir, Heterogeneity_Adj = False, Heterogeneity_Adj_Moc = False ):

    if Heterogeneity_Adj: 
        pd_pre_moc = pd.read_csv(seg_dir+'\\MOC\\Hetero_Adj.csv', low_memory=False).rename(columns = {'Baseline PD': col_dft})
    else: 
        pd_pre_moc = df_trn.groupby([col_seg, col_dt], as_index=False)[col_dft].mean()
        pd_pre_moc = pd_pre_moc.groupby([col_seg], as_index=False)[col_dft].mean()
    
    mdl_param = apply_moc_2_pd(pd_pre_moc, file_moc_total_pkl, col_pd=col_dft, Heterogeneity_Adj_Moc= Heterogeneity_Adj_Moc)
    with open(file_mdl_pkl, 'wb') as file:
        pickle.dump(mdl_param, file)
    return mdl_param

#%%
def apply_param_pd(df_sp, col_seg, col_param_pd, file_mdl_pkl, file_out_csv=None):
    with open(file_mdl_pkl, 'rb') as file:
        df_mdl = pickle.load(file)
    df_sp = df_sp.merge(df_mdl[[col_seg]+ col_param_pd], on=col_seg, how='left')
    if file_out_csv is not None:
        df_sp.to_csv(file_out_csv, index=False)
    return df_sp

