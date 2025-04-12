def rwa_calculation(df_input_data, lgd_gen_floor, CMHC_lgd, CMHC_pd):

    """
    rwa_calculation(df_input_data, adhoc_path, lgd_gen_floor, CMHC_lgd, CMHC_pd)
    This function calculates the Risk-Weighted Assets (RWA) for residential mortgages 
    based on various regulatory and business rules. It processes raw data, applies 
    adjustments, and computes RWA values for different insured and uninsured loan 
    categories. The function also generates summary tables and outputs detailed 
    results for further analysis.
    Parameters:
        df_input_data (pd.DataFrame): The raw input data containing loan-level 
            information, including PD, LGD, EAD, and other attributes.
        lgd_gen_floor (float): The general floor value for LGD as per regulatory 
            requirements.
        CMHC_lgd (float): The LGD value to be applied for CMHC-insured loans.
        CMHC_pd (float): The PD value to be applied for CMHC-insured loans.
    Returns:
        tuple: A tuple containing the following:
            - df_out (pd.DataFrame): A summary DataFrame with aggregated RWA 
              values for different scenarios (e.g., pre-MOC, post-MOC).
            - rwa_by_Insured_class (pd.DataFrame): A DataFrame summarizing RWA 
              values grouped by insured class (e.g., CMHC, Sagen, CG, Uninsured).
            - rwa_by_MRS_Bin (pd.DataFrame): A DataFrame summarizing RWA values 
              grouped by MRS Bin PD.
            - res_data (pd.DataFrame): The processed loan-level data with 
              calculated RWA values and intermediate results.
    Notes:
        - The function applies different LGD and PD adjustments based on the 
          insured class of the loans (e.g., CMHC, Sagen, CG, Uninsured).
        - Regulatory floors and maturity adjustments are applied to ensure 
          compliance with CAR 2023 guidelines.
        - The function uses both retail and corporate risk weight formulas 
          depending on the loan type and insurance status.
        - The function generates summary tables for pre-MOC, post-MOC, and final 
          RWA values, along with breakdowns by insured class and MRS Bin PD.
    """


    import os
    import pandas as pd
    import numpy as np
        

    df_out=pd.DataFrame()

    # record result data 
    res_data=pd.DataFrame()


    for ind_contol in range(0,3):
        print(ind_contol)

        # baseline LGD, Baseline PD
        if ind_contol==0:
            rwa_raw_data = df_input_data.copy().drop(['PD_Post_MOC','PD_Post_MOC_Pre_Adj'], axis=1).rename({'PD_Pre_MOC': 'CalibratedPD', 'Base_Line_LGD' : 'Pre_final_LGD'  }, axis='columns'); 
        #  PD_Post_MOC, LGD_DT_JUST     
        if ind_contol==1:
            rwa_raw_data = df_input_data.copy().drop(['PD_Pre_MOC','PD_Post_MOC_Pre_Adj'], axis=1).rename({'PD_Post_MOC': 'CalibratedPD', 'LGD_DT_Adjusted': 'Pre_final_LGD' }, axis='columns'); 
        #  PD_Post_MOC, Final LGD     
        if ind_contol==2:
            rwa_raw_data = df_input_data.copy().drop(['PD_Pre_MOC','PD_Post_MOC_Pre_Adj'], axis=1).rename({'PD_Post_MOC': 'CalibratedPD' , 'Model_LGD': 'Pre_final_LGD' }, axis='columns'); 
            
        #  aseline LGD,  PD after long-run adjustment
        '''
        if ind_contol==3:
            rwa_raw_data = df_input_data.copy().drop(['PD_Post_MOC','PD_Post_MOC_Pre_Adj'], axis=1).rename({'PD_Pre_MOC': 'CalibratedPD', 'Base_Line_LGD' : 'Pre_final_LGD'  }, axis='columns'); 
            
            adj_Longrun = pd.read_excel(os.path.join(adhoc_path, 'MOC_Summary.xlsx'), sheet_name= 'Sheet1') [['Adj_Longrun']].values[0].item()
            rwa_raw_data['CalibratedPD'] =  rwa_raw_data['CalibratedPD']*(1+adj_Longrun)
        '''
        
        # Apply regulatory floors and derive the final LGD
        rwa_raw_data['DLGD_floor'] = rwa_raw_data['Segment_Avg_LGD'] + rwa_raw_data['AddOn']
        
        # sum(~rwa_raw_data['DLGD_floor'].isna())
        # sum((~rwa_raw_data['Segment_Avg_LGD'].isna()) & (rwa_raw_data.Insured_class.isin(["CMHC","Sagen", "CG"])))
        
        # sum(~rwa_raw_data['Pre_final_LGD'].isna())
        # sum(~rwa_raw_data['Segment_Avg_LGD'].isna() )
        # sum(~(rwa_raw_data.Insured_class.isin(["CMHC","Sagen", "CG"]) ))

        rwa_raw_data.loc[~rwa_raw_data['Pre_final_LGD'].isna(),'OSFI_LGD_floor'] = lgd_gen_floor
        rwa_raw_data['Final_LGD'] = rwa_raw_data[['Pre_final_LGD','DLGD_floor','OSFI_LGD_floor']].max(axis='columns')

        # sum(~rwa_raw_data['Final_LGD'].isna())
        
        #for insured loans which do not have Gen3 LGD, previous Final LGD is used
        #rwa_raw_data.loc[rwa_raw_data['Pre_final_LGD'].isna(),'Final_LGD'] = rwa_raw_data['Final_LGD_old']
        
        
        ## CG and Sagen use 11% LGD 
        rwa_raw_data.loc[rwa_raw_data.Insured_class.isin(["CMHC","Sagen", "CG"]), 'Final_LGD' ] = CMHC_lgd

        # sum(rwa_raw_data['Final_LGD'].isna())
        
        
        ## insured_PMI_ratio 

        
        rwa_raw_data["deductible_amount"] = 0.1 * rwa_raw_data.Advance_Amount
        rwa_raw_data["insured_PMI_ratio"] = np.where(rwa_raw_data.EAD > 0, (rwa_raw_data.EAD - rwa_raw_data.deductible_amount) / rwa_raw_data.EAD, 0)
        
        # !!!! can be negative; When insured_PMI_ratio is negative it will be handle in cond_PMI_dec case.
        
        
         #  sum( round(rwa_raw_data["insured_PMI_ratio"]*10**5) == round(rwa_raw_data["insured_PMI_ratio_old"] *10**5) ) #56478
        
        ## Generate indicators 
        cond_CMHC = (rwa_raw_data.Insured_class.isin(['CMHC']))
        
        cond_PMI = (rwa_raw_data.Insured_class.isin(["Sagen", "CG"])) & (rwa_raw_data.EAD > 0.1 * rwa_raw_data.Advance_Amount)

        cond_PMI_dec = (rwa_raw_data.Insured_class.isin(["Sagen", "CG"])) & (rwa_raw_data.EAD <= 0.1 * rwa_raw_data.Advance_Amount)
        
        cond_UI = (rwa_raw_data.Insured_class.isin(["Uninsured"]))
        
        
        ## Maturity adjustment
        
        def func_maturity_adj(pd_value):
            import numpy as np
            adj = (0.11852 - 0.05478 * np.log(pd_value)) ** 2
            return adj  ##CAR 2023 5.3.1(i) 66.
        
        rwa_raw_data.loc[cond_UI | cond_PMI_dec, "Maturity_adj"] = rwa_raw_data.loc[:, ['CalibratedPD']].apply(lambda s: func_maturity_adj(s[0]), axis=1) #not actually used since retail RWA formula will be applied
        
        rwa_raw_data.loc[cond_CMHC | cond_PMI, "Maturity_adj"] = func_maturity_adj(CMHC_pd)
        

        ## 'corr_uninsured'  depends on rental income indicator and has two values correlation_residential_mortgages_rental,correlation_residential_mortgages
        
        
        # correlation for CMHC and insured deductible for PMI ?
        def correlation_corp(pd_value):
            import numpy as np
            #     corr =  round(0.12 * (1 - np.exp(-50 * pd_value)) / (1- np.exp(-50)) +
            #                       0.24 * (1 - (1- np.exp(-50 * pd_value))/(1 - np.exp(-50))),6) ##CAR 2023 5.3.1(i) 66.
            corr = 0.12 * (1 - np.exp(-50 * pd_value)) / (1 - np.exp(-50)) + 0.24 * (
                    1 - (1 - np.exp(-50 * pd_value)) / (1 - np.exp(-50)))
            return corr
        
        rwa_raw_data.loc[cond_CMHC | cond_PMI, 'corr_insured'] = correlation_corp(CMHC_pd)
        
        
        ## func_risk_weight_retail 
        def func_risk_weight_retail(pd_value, lgd_value, corr):  # CAR 2023 Chapter 5 paragrah 81
            import scipy.stats as st
            pd_value = float(pd_value)
            lgd_value = float(lgd_value)
            corr = float(corr)
            pd_UL = st.norm.cdf(
                (1 - corr) ** (-0.5) * st.norm.ppf(pd_value) + (corr / (1 - corr)) ** (0.5) * st.norm.ppf(0.999))
            K = pd_UL * lgd_value - pd_value * lgd_value
            # return  round(pd_UL * lgd_value - pd_value * lgd_value,6)
            return K
        
        
        def func_risk_weight_corp(pd_value, lgd_value, corr, maturity_adj,
                                effective_maturity):  # CAR 2023 Chapter 5 paragraph 66
            import scipy.stats as st
            a = (lgd_value * st.norm.cdf((1 - corr) ** (-0.5) * st.norm.ppf(pd_value) +
                                        (corr / (1 - corr)) ** 0.5 * st.norm.ppf(0.999)))
            tail_value = a - (pd_value * lgd_value)
            adj = (1 + (effective_maturity - 2.5) * maturity_adj) / (1 - (1.5 * maturity_adj))
            k = tail_value * adj
            # return  round(tail_value * adj,6)
            return k
        
        
        # retail risk weight function for uninsured loans
        rwa_raw_data['risk_weight_uninsured'] = 0
        
        rwa_raw_data.loc[cond_UI, 'risk_weight_uninsured'] = rwa_raw_data.loc[cond_UI, ['CalibratedPD', 'Final_LGD', 'corr_uninsured']].apply(
                lambda s: func_risk_weight_retail(s[0], s[1], s[2]), axis=1)
        
        
        # re-write uninsured deductible for PMI using borrower PD and 100% LGD and retail risk weight function
        # df["LGD_uninsured"]=1.0
        
        rwa_raw_data.loc[cond_PMI, 'risk_weight_uninsured'] = rwa_raw_data.loc[cond_PMI, ['CalibratedPD', 'corr_uninsured']].apply(
                lambda s: func_risk_weight_retail(s[0], 1, s[1]), axis=1)
        
        rwa_raw_data.loc[cond_PMI_dec, 'risk_weight_uninsured'] = rwa_raw_data.loc[cond_PMI_dec, ['CalibratedPD', 'corr_uninsured']].apply(
                lambda s: func_risk_weight_retail(s[0], 1, s[1]), axis=1)
        
            
        # insured CMHC including PMI insured backstop use non-retail risk function
        rwa_raw_data['risk_weight_insured'] = 0
        
        rwa_raw_data.loc[cond_CMHC, 'risk_weight_insured'] = rwa_raw_data.loc[cond_CMHC, ['Final_LGD', 'corr_insured','Maturity_adj', 'Years_to_maturity']].apply(
            lambda s: func_risk_weight_corp(CMHC_pd, CMHC_lgd, s[1], s[2], s[3]), axis=1)
        
        
        
        rwa_raw_data.loc[cond_PMI, 'risk_weight_insured'] = rwa_raw_data.loc[cond_PMI, ['Final_LGD', 'corr_insured','Maturity_adj', 'Years_to_maturity']].apply(
            lambda s: func_risk_weight_corp(CMHC_pd,  CMHC_lgd, s[1], s[2], s[3]), axis=1)
        
        
        
        ## RWA_res_mortgage 
        def RWA_res_mortgage(capital_value, capital_value_insured, EAD_value, insured_ratio):
            # insured_ratio: Ratio of insured
            #     insured_RWA = round(capital_value_insured * 12.5 * EAD_value,0)
            #     uninsured_RWA = round(capital_value * 12.5 * EAD_value,0)
            insured_RWA = capital_value_insured * 12.5 * EAD_value
            uninsured_RWA = capital_value * 12.5 * EAD_value
            rwa_total = insured_ratio * insured_RWA + (1 - insured_ratio) * uninsured_RWA
            return rwa_total
        
        
        
        # 1. CMHC insured
        rwa_raw_data.loc[cond_CMHC, 'RWA_new'] = rwa_raw_data.loc[cond_CMHC, [ 'risk_weight_insured', 'EAD']].apply(lambda s: RWA_res_mortgage(0, s[0], s[1], 1), axis=1)
        
        
        
        # 2. Sagen and CG
        rwa_raw_data.loc[cond_PMI, 'RWA_new'] = rwa_raw_data.loc[
            cond_PMI, ['risk_weight_uninsured', 'risk_weight_insured', 'EAD', "insured_PMI_ratio"]].apply(lambda s: RWA_res_mortgage(s[0], s[1], s[2], s[3]), axis=1)

        rwa_raw_data.loc[cond_PMI_dec, 'RWA_new'] = rwa_raw_data.loc[cond_PMI_dec, ['risk_weight_uninsured', 'EAD']].apply(
            lambda s: RWA_res_mortgage(s[0], 0, s[1], 0), axis=1)
        
        # 3. Uninsured Loans
        rwa_raw_data.loc[cond_UI, 'RWA_new'] = rwa_raw_data.loc[cond_UI, ['risk_weight_uninsured', 'EAD']].apply(
            lambda s: RWA_res_mortgage(s[0], 0, s[1], 0), axis=1)
        
        #test = rwa_raw_data.loc[ (round(rwa_raw_data["rwa_airb"]*10**6) != round(rwa_raw_data["RWA_new"]*10**6) )  &  (rwa_raw_data["Insured_class"] == 'CMHC') ][['Insured_class',"rwa_airb",'RWA_new']]
        
        #rwa_raw_data.loc[rwa_raw_data['Insured_class'] =='Uninsufred', 'RWA_new'].sum()
        ## Summary table
        
        if ind_contol==0:
            df_out["EAD"]= [rwa_raw_data["EAD"].sum()]
            # df_out["SA RWA"]= [rwa_raw_data["RWA_standardized"].sum()]  
            # df_out["In Product RWA"]= [rwa_raw_data["rwa_airb"].sum()]  --gen 2 
            df_out["Pre MOC RWA"]= [rwa_raw_data["RWA_new"].sum()]
            
            df_out["Pre MOC RWA (uninsured)"]= [rwa_raw_data.loc[rwa_raw_data['Insured_class']=='Uninsured']["RWA_new"].sum()]
            df_out["EAD (uninsured)"]= [rwa_raw_data.loc[rwa_raw_data['Insured_class']=='Uninsured']["EAD"].sum()]

            
            # rwa_airb_by_Insured_class = rwa_raw_data.groupby(['Insured_class' ],as_index=False)[['EAD','rwa_airb','RWA_standardized']].sum().rename({'rwa_airb': 'rwa_airb_by_Insured_class', 'RWA_standardized': 'RWA_standardized_by_Insured_class'}, axis='columns')
            
            rwa_mapped_by_Insured_class = rwa_raw_data.groupby(['Insured_class' ],as_index=False)[['EAD','RWA_new']].sum().rename({'RWA_new': 'rwa_mapped_by_Insured_class'}, axis='columns')
            
            # rwa_airb_by_MRS_Bin = rwa_raw_data.groupby(['MRS_Bin_PD' ],as_index=False)[['EAD','rwa_airb','RWA_standardized']].sum().rename({'rwa_airb': 'rwa_airb_by_MRS_Bin'}, axis='columns')
            
            rwa_mapped_by_MRS_Bin = rwa_raw_data.groupby(['MRS_Bin_PD' ],as_index=False)[['EAD','RWA_new']].sum().rename({'RWA_new': 'rwa_mapped_by_MRS_Bin'#,'RWA_standardized': 'RWA_standardized_by_MRS_Bin'
                                                                                                                }, axis='columns')
            
            # rwa_by_Insured_class0 =pd.merge(rwa_airb_by_Insured_class, rwa_mapped_by_Insured_class, how='left',on=['Insured_class'] )
            
            # rwa_by_MRS_Bin0 =pd.merge(rwa_airb_by_MRS_Bin, rwa_mapped_by_MRS_Bin, how='left',on=['MRS_Bin_PD'] )
        
        if ind_contol==1: 
            df_out["Post MOC Pre LR RWA"]= [rwa_raw_data["RWA_new"].sum()]
            df_out["Post MOC Pre LR RWA (uninsured)"]= [rwa_raw_data.loc[rwa_raw_data['Insured_class']=='Uninsured']["RWA_new"].sum()]
        
        '''   
        if ind_contol==3: 
            df_out["Post PD longrun"]= [rwa_raw_data["RWA_new"].sum()]
            df_out["Post PD longrun (uninsured)"]= [rwa_raw_data.loc[rwa_raw_data['Insured_class']=='Uninsured']["RWA_new"].sum()]    
        ''' 
            
        if ind_contol==2: 
            df_out["Final RWA"]= [rwa_raw_data["RWA_new"].sum()]

            rwa_Post_MOC_by_Insured_class = rwa_raw_data.groupby(['Insured_class' ],as_index=False)['RWA_new'].sum().rename({'RWA_new': 'rwa_Post_MOC_by_Insured_class'}, axis='columns')

            rwa_Post_MOC_by_MRS_Bin = rwa_raw_data.groupby(['MRS_Bin_PD'],as_index=False)['RWA_new'].sum().rename({'RWA_new': 'rwa_Post_MOC_by_MRS_Bin'}, axis='columns')

            # rwa_by_Insured_class =pd.merge(rwa_by_Insured_class0, rwa_Post_MOC_by_Insured_class, how='left',on=['Insured_class'] )
            rwa_by_Insured_class =pd.merge( rwa_mapped_by_Insured_class, rwa_Post_MOC_by_Insured_class, how='left',on=['Insured_class'] )

            # rwa_by_MRS_Bin =pd.merge(rwa_by_MRS_Bin0, rwa_Post_MOC_by_MRS_Bin, how='left',on=['MRS_Bin_PD'] )
            rwa_by_MRS_Bin =pd.merge(rwa_mapped_by_MRS_Bin, rwa_Post_MOC_by_MRS_Bin, how='left',on=['MRS_Bin_PD'] )

            #copy to res_data
            res_data = rwa_raw_data.copy()

    return df_out, rwa_by_Insured_class, rwa_by_MRS_Bin, res_data