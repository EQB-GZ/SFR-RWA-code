def func_execution_rwa(col_pre_pd,col_pre_lgd,mrs_ind,col_deductible,output_grp):
    import numpy as np
    import pandas as pd
    from sfr02_data_cleaning import df

    # AIRB RWA 2023 Calculation

    from func_airb_rwa import airb_rwa_calc_2023_retail,airb_rwa_calc_2023_corp,airb_rwa_calc_2023_corp_v2
    from sfr00_param_controls import pd_substitution

    if mrs_ind:
        df=airb_rwa_calc_2023_retail(df, pd_substitution,
                                  col_in=['Calibrated' + col_pre_pd, 'Final_LGD', 'EAD', col_deductible],
                                  col_out=['corr_uninsured', 'corr_insured', 'Capital_CAR', 'Capital_CAR_insured',
                                           'RWA_AIRB'])
        df_corp=airb_rwa_calc_2023_corp_v2(df, pd_substitution,
                                   col_in=['Calibrated' + col_pre_pd, 'Final_LGD', 'EAD', col_deductible,
                                           'Advance_Amount'],
                                   col_out=['corr_uninsured_corp', 'corr_insured_corp', 'risk_weight_uninsured',
                                            'risk_weight_insured', 'RWA_AIRB_corp']
                                   )
    else:
        df=airb_rwa_calc_2023_retail(df, pd_substitution, col_in=[col_pre_pd, 'Final_LGD', 'EAD', col_deductible],
                                  col_out=['corr_uninsured', 'corr_insured', 'Capital_CAR', 'Capital_CAR_insured',
                                           'RWA_AIRB'])
        df_corp=airb_rwa_calc_2023_corp_v2(df,pd_substitution,
                                   col_in=[col_pre_pd, 'Final_LGD', 'EAD', col_deductible,'Advance_Amount'],
                                   col_out=['corr_uninsured_corp', 'corr_insured_copr', 'risk_weight_uninsured',
                                             'risk_weight_insured','RWA_AIRB']
                                   )
    # col_airb_rwa_out=['Loan_Number','Insured_class','corr_CAR','Capital_CAR','Capital_CAR_insured','RWA_CAR']

    # AIRB RWA 2019 Calculation

    from func_airb_rwa import airb_rwa_calc_2019

    if mrs_ind:
        df=airb_rwa_calc_2019(df, pd_substitution, col_in=['Calibrated' + col_pre_pd, 'Final_LGD', 'EAD'],
                           col_out=['corr_CAR_2019', 'Capital_CAR_2019', 'Capital_CAR_insured_2019', 'RWA_AIRB_2019'])
    else:
        df=airb_rwa_calc_2019(df, pd_substitution, col_in=[col_pre_pd, 'Final_LGD', 'EAD'],
                           col_out=['corr_CAR_2019', 'Capital_CAR_2019', 'Capital_CAR_insured_2019', 'RWA_AIRB_2019'])

    # Standardized RWA 2023 Calculation

    from func_stand_rwa import stand_rwa_calc_2023, stand_rwa_calc_2019

    # stand_rwa_calc_2023(df,cols_in=['Current_LTV','EAD'],cols_out=['BCAR_weight_adj','RWA_standardized'])

    df=stand_rwa_calc_2023(df, col_pd='PDResult',col_rental='Rental_Income',col_insured='Insured_class',
                        cols_in=['Current_LTV', 'EAD', 'insured_PMI_ratio','Advance_Amount'],
                        cols_out=['BCAR_weight_adj', 'RWA_standardized']
                        )

    col_stand_rwa_out = ['Loan_Number', 'Insured_class', 'RWA_standardized']

    # Standardized RWA 2019 Calculation
    # df['BCAR_Weighting'] = df['BCAR_Weighting'].astype(float)
    df=stand_rwa_calc_2019(df, col_pd='PDResult',col_insured='Insured_class',
                        cols_in=['BCAR_Weighting', 'Current_LTV', 'EAD'],
                        cols_out=['BCAR_weight_adj_2019', 'RWA_standardized_2019']
                        )

    df['Portfolio'] = np.where(df.Insured_class == 'Uninsured', 'Uninsured', 'Insured')  # add sub portfolio
    df_report = df.loc[(df['RiskRating_PD'] != 'Defaulted') & (df['PDResult'] != 1.0), :]  # remove default loans

    # Summary results
    from func_output import rwa_summary

    if mrs_ind:
        summary_tbl = rwa_summary(df_report, output_grp, col_pd="Calibrated" + col_pre_pd)
    else:
        summary_tbl = rwa_summary(df_report, output_grp, col_pd=col_pre_pd)

    return df,df_report, summary_tbl,df_corp

