def airb_rwa_calc_2019(df, pd_substitution, col_in=None,
                       col_out=None):
    # import pandas as pd
    # import scipy.stats as st
    if col_in is None:
        col_in = ['Final_PD', 'Final_LGD', 'EAD']
    if col_out is None:
        col_out = ['corr_2019', 'risk_weight_uninsured_2019', 'risk_weight_insured_2019', 'RWA_2019']
    from myFunctions import func_risk_weight_retail, RWA_res_mortgage
    from sfr00_param_controls import correlation_residential_mortgages
    cond_CMHC = df.Insured_class.isin(['CMHC'])
    cond_PMI = df.Insured_class.isin(["Sagen", "CG"])
    cond_UI = df.Insured_class.isin(["Uninsured"])

    borrower_pd = col_in[0]
    borrower_lgd = col_in[1]
    exposure = col_in[2]
    correlation = col_out[0]
    risk_weight_uninsured = col_out[1]
    risk_weight_insured = col_out[2]
    rwa = col_out[3]

    df.loc[:, correlation] = correlation_residential_mortgages  # correlation
    df.loc[:, risk_weight_uninsured] = df.loc[:, [borrower_pd, borrower_lgd, correlation]].apply(
        lambda s: func_risk_weight_retail(s[0], s[1], s[2]), axis=1)  # risk weight function
    df.loc[:, risk_weight_insured] = df.loc[:, [borrower_lgd, correlation]].apply(
        lambda s: func_risk_weight_retail(pd_substitution, s[0], s[1]),
        axis=1)  # risk weight function for insured portion

    # 1. CMHC insured
    df.loc[cond_CMHC, rwa] = df.loc[cond_CMHC, [risk_weight_insured, exposure]].apply(
        lambda s: RWA_res_mortgage(0, s[0], s[1], 1), axis=1)
    # 2. Sagen and CG
    df.loc[cond_PMI, rwa] = df.loc[cond_PMI, [risk_weight_uninsured, risk_weight_insured, exposure]].apply(
        lambda s: RWA_res_mortgage(s[0], s[1], s[2], 0.9), axis=1)
    # 3. Uninsured Loans
    df.loc[cond_UI, rwa] = df.loc[cond_UI, [risk_weight_uninsured, exposure]].apply(
        lambda s: RWA_res_mortgage(s[0], 0, s[1], 0), axis=1)
    return df


# def airb_rwa_calc_2023(df, pd_substitution, col_in=None,
#                        col_out=None):
#     if col_out is None:
#         col_out = ['corr_uninsured', 'corr_insured', 'Capital_CAR', 'Capital_CAR_insured', 'RWA_CAR']
#     if col_in is None:
#         col_in = ['MRS_pd_test', 'Final_LGD', 'EAD', 'insured_PMI_ratio']
#     import pandas as pd
#     import numpy as np
#     # import scipy.stats as st
#     from myFunctions import func_risk_weight_retail, func_risk_weight_corp, RWA_res_mortgage, maturity_adj, \
#         correlation_corp
#     from sfr_param_controls import correlation_residential_mortgages, correlation_residential_mortgages_rental, dti_raw
#
#     borrower_pd = col_in[0]
#     borrower_lgd = col_in[1]
#     exposure = col_in[2]
#     deductible_ratio = col_in[3]
#     corr_uninsured = col_out[0]
#     corr_insured = col_out[1]
#     risk_weight_uninsured = col_out[2]
#     risk_weight_insured = col_out[3]
#     rwa = col_out[4]
#
#     cond_CMHC = (df.Insured_class.isin(['CMHC']))
#     # cond_PMI = (df.Insured_class.isin(["Sagen","CG"])) & (df.Remaining_Principal_Post_CRM > 0.1*df.Advance_Amount)
#     # cond_UI = (df.Insured_class.isin(["Uninsured"])) | ((df.Insured_class.isin(["Sagen","CG"])) & (
#     # df.Remaining_Principal_Post_CRM <= 0.1*df.Advance_Amount))
#
#     cond_PMI = (df.Insured_class.isin(["Sagen", "CG"])) & (df.EAD > 0.1 * df.Advance_Amount)
#     cond_UI = (df.Insured_class.isin(["Uninsured"])) | (
#             (df.Insured_class.isin(["Sagen", "CG"])) & (df.EAD <= 0.1 * df.Advance_Amount))
#     cond_rental = (df.Rental_Income == 'Y')
#
#     df['Maturity_Date'] = pd.to_datetime(df['Maturity_Date'])
#     dti = pd.to_datetime(dti_raw)
#     df.loc[:, "Days_to_maturity"] = (df.Maturity_Date - dti).dt.days
#     df.loc[:, "Years_to_maturity"] = df.loc[:, "Days_to_maturity"] / 365.25
#     df['Years_to_maturity'] = np.where(df.Years_to_maturity <= 1, 1,
#                                        np.where(df.Years_to_maturity >= 5, 5,
#                                                 df.Years_to_maturity))
#     df.loc[:, "Maturity_adj"] = df.loc[:, [borrower_pd]].apply(lambda s: maturity_adj(s[0]), axis=1)
#
#     df.loc[:, corr_uninsured] = np.where(cond_rental, correlation_residential_mortgages_rental,
#                                          correlation_residential_mortgages)
#     df.loc[:, corr_insured] = np.where(cond_rental, correlation_residential_mortgages_rental,
#                                        correlation_residential_mortgages)
#
#     # re-write correlation for insured deductible for PMI ?
#     df.loc[cond_PMI, corr_insured] = df.loc[:, [borrower_pd]].apply(lambda s: correlation_corp(s[0]), axis=1)
#
#     # df.loc[:,col_out[1]] = df.loc[:,[col_in[0],col_in[1],col_out[0]]].apply(lambda s: Capital_CAR(s[0],s[1],s[2]),
#     # axis =1)
#
#     # retail risk weight function for uninsured loans
#     df[risk_weight_uninsured] = 0
#     df.loc[cond_UI, risk_weight_uninsured] = df.loc[:, [borrower_pd, borrower_lgd, corr_uninsured]].apply(
#         lambda s: func_risk_weight_retail(s[0], s[1], s[2]), axis=1)
#
#     # re-write uninsured deductible for PMI using borrower PD and 100% LGD and retail risk weight function
#     df.loc[cond_PMI, risk_weight_uninsured] = df.loc[:, [borrower_pd, corr_uninsured]].apply(
#         lambda s: func_risk_weight_retail(s[0], 1, s[1]), axis=1)
#     # df.loc[cond_PMI,col_out[1]] = df.loc[:,[col_in[0],col_out[1],'Maturity_adj',"Years_to_maturity"] ].apply(lambda
#     # s: func_risk_weight_corp(s[0],1,s[1],s[2],s[3]), axis =1 )
#
#     # insured CMHC including PMI insured backstop use corporate risk function
#     df[risk_weight_insured] = 0
#     df.loc[cond_CMHC, risk_weight_insured] = df.loc[:, [borrower_lgd, corr_insured]].apply(
#         lambda s: func_risk_weight_retail(pd_substitution, s[0], s[1]), axis=1)
#     df.loc[cond_PMI, risk_weight_insured] = df.loc[:,
#                                             [borrower_lgd, corr_insured, 'Maturity_adj', "Years_to_maturity"]].apply(
#         lambda s: func_risk_weight_corp(pd_substitution, s[0], s[1], s[2], s[3]), axis=1)
#
#     # 1. CMHC insured
#     df.loc[cond_CMHC, rwa] = df.loc[cond_CMHC, [risk_weight_insured, exposure]].apply(
#         lambda s: RWA_res_mortgage(0, s[0], s[1], 1), axis=1)
#
#     # 2. Sagen and CG
#     df.loc[cond_PMI, rwa] = df.loc[
#         cond_PMI, [risk_weight_uninsured, risk_weight_insured, exposure, deductible_ratio]].apply(
#         lambda s: RWA_res_mortgage(s[0], s[1], s[2], s[3]), axis=1)
#
#     # 3. Uninsured Loans
#     df.loc[cond_UI, rwa] = df.loc[cond_UI, [risk_weight_uninsured, exposure]].apply(
#         lambda s: RWA_res_mortgage(s[0], 0, s[1], 0), axis=1)
#
#     return df


def airb_rwa_calc_2023_retail(df, CMHC_pd, col_in=None,
                              col_out=None):
    if col_out is None:
        col_out = ['corr_uninsured', 'corr_insured', 'risk_weight_uninsured', 'risk_weight_insured',
                   'RWA']
    if col_in is None:
        col_in = ['Final_PD', 'Final_LGD', 'EAD', 'insured_PMI_ratio']
    # import pandas as pd
    import numpy as np
    # import scipy.stats as st
    from myFunctions import func_risk_weight_retail,  RWA_res_mortgage
    from sfr00_param_controls import correlation_residential_mortgages, correlation_residential_mortgages_rental

    borrower_pd = col_in[0]
    borrower_lgd = col_in[1]
    exposure = col_in[2]
    deductible_ratio = col_in[3]
    corr_uninsured = col_out[0]
    corr_insured = col_out[1]
    risk_weight_uninsured = col_out[2]
    risk_weight_insured = col_out[3]
    rwa = col_out[4]

    cond_CMHC = (df.Insured_class.isin(['CMHC']))
    cond_PMI = (df.Insured_class.isin(["Sagen", "CG"])) & (df.EAD > 0.1 * df.Advance_Amount)
    cond_UI = (df.Insured_class.isin(["Uninsured"])) | (
            (df.Insured_class.isin(["Sagen", "CG"])) & (df.EAD <= 0.1 * df.Advance_Amount))
    cond_rental = (df.Rental_Income == 'Y')

    df.loc[:, corr_uninsured] = np.where(cond_rental, correlation_residential_mortgages_rental,
                                         correlation_residential_mortgages)
    df.loc[:, corr_insured] = np.where(cond_rental, correlation_residential_mortgages_rental,
                                       correlation_residential_mortgages)

    # retail risk weight function for uninsured loans
    df[risk_weight_uninsured] = 0
    df.loc[cond_UI, risk_weight_uninsured] = df.loc[cond_UI, [borrower_pd, borrower_lgd, corr_uninsured]].apply(
        lambda s: func_risk_weight_retail(s[0], s[1], s[2]), axis=1)

    # re-write uninsured deductible for PMI using borrower PD and 100% LGD and retail risk weight function
    # df["LGD_uninsured"]=1.0
    df.loc[cond_PMI, risk_weight_uninsured] = df.loc[cond_PMI, [borrower_pd, corr_uninsured]].apply(
        lambda s: func_risk_weight_retail(s[0], 1, s[1]), axis=1)

    # insured CMHC including PMI insured backstop use retail risk function
    df[risk_weight_insured] = 0
    df.loc[cond_CMHC, risk_weight_insured] = df.loc[cond_CMHC, [borrower_lgd, corr_insured]].apply(
        lambda s: func_risk_weight_retail(CMHC_pd, s[0], s[1]), axis=1)
    df.loc[cond_PMI, risk_weight_insured] = df.loc[cond_PMI, [borrower_lgd, corr_insured]].apply(
        lambda s: func_risk_weight_retail(CMHC_pd, s[0], s[1]), axis=1)

    # 1. CMHC insured
    df.loc[cond_CMHC, rwa] = df.loc[cond_CMHC, [risk_weight_insured, exposure]].apply(
        lambda s: RWA_res_mortgage(0, s[0], s[1], 1), axis=1)

    # 2. Sagen and CG
    df.loc[cond_PMI, rwa] = df.loc[
        cond_PMI, [risk_weight_uninsured, risk_weight_insured, exposure, deductible_ratio]].apply(
        lambda s: RWA_res_mortgage(s[0], s[1], s[2], s[3]), axis=1)

    # 3. Uninsured Loans
    df.loc[cond_UI, rwa] = df.loc[cond_UI, [risk_weight_uninsured, exposure]].apply(
        lambda s: RWA_res_mortgage(s[0], 0, s[1], 0), axis=1)

    return df


def airb_rwa_calc_2023_corp(df, CMHC_pd, col_in=None,
                            col_out=None):
    if col_in is None:
        col_in = ['CalibratedPD', 'Final_LGD', 'EAD', 'insured_PMI_ratio', 'Advance_Amount']
    if col_out is None:
        col_out = ['corr_uninsured', 'corr_insured', 'risk_weight_uninsured', 'risk_weight_insured',
                   'rwa']
    import pandas as pd
    import numpy as np
    # import scipy.stats as st
    from myFunctions import func_risk_weight_retail, func_risk_weight_corp, RWA_res_mortgage, func_maturity_adj, \
        correlation_corp
    from sfr00_param_controls import correlation_residential_mortgages, correlation_residential_mortgages_rental, \
        dti_raw

    borrower_pd = col_in[0]
    borrower_lgd = col_in[1]
    exposure = col_in[2]
    deductible_ratio = col_in[3]
    origination_amount = col_in[4]

    corr_uninsured = col_out[0]
    corr_insured = col_out[1]
    risk_weight_uninsured = col_out[2]
    risk_weight_insured = col_out[3]
    rwa = col_out[4]

    cond_CMHC = (df.Insured_class.isin(['CMHC']))
    cond_PMI = (df.Insured_class.isin(["Sagen", "CG"])) & (df[exposure] > 0.1 * df[origination_amount])
    cond_UI = (df.Insured_class.isin(["Uninsured"])) | (
            (df.Insured_class.isin(["Sagen", "CG"])) & (df[exposure] <= 0.1 * df[origination_amount]))
    cond_rental = (df.Rental_Income == 'Y')

    df['Maturity_Date'] = pd.to_datetime(df['Maturity_Date'])
    dti = pd.to_datetime(dti_raw)
    df.loc[:, "Days_to_maturity"] = (df.Maturity_Date - dti).dt.days
    df.loc[:, "Years_to_maturity"] = df.loc[:, "Days_to_maturity"] / 365.25
    df['Years_to_maturity'] = np.where(df.Years_to_maturity <= 1, 1,
                                       np.where(df.Years_to_maturity >= 5, 5,
                                                df.Years_to_maturity))
    df.loc[:, "Maturity_adj"] = df.loc[:, [borrower_pd]].apply(lambda s: func_maturity_adj(s[0]), axis=1)

    df.loc[:, corr_uninsured] = np.where(cond_rental, correlation_residential_mortgages_rental,
                                         correlation_residential_mortgages)

    # correlation for CMHC and insured deductible for PMI ?
    df.loc[cond_CMHC, corr_insured] = df.loc[:, [borrower_pd]].apply(lambda s: correlation_corp(s[0]), axis=1)
    df.loc[cond_PMI, corr_insured] = df.loc[:, [borrower_pd]].apply(lambda s: correlation_corp(s[0]), axis=1)

    ##### retail risk weight function for uninsured loans
    df[risk_weight_uninsured] = 0
    df.loc[cond_UI, risk_weight_uninsured] = df.loc[:, [borrower_pd, borrower_lgd, corr_uninsured]].apply(
        lambda s: func_risk_weight_retail(s[0], s[1], s[2]), axis=1)

    ####### re-write uninsred deductable for PMI using borrower PD and 100% LGD and retail risk weight function
    df.loc[cond_PMI, risk_weight_uninsured] = df.loc[:, [borrower_pd, corr_uninsured]].apply(
        lambda s: func_risk_weight_retail(s[0], 1, s[1]), axis=1)

    #### insured CMHC including PMI insured backstop use corporaterisk function
    df[risk_weight_insured] = 0

    df.loc[cond_CMHC, risk_weight_insured] = df.loc[:,
                                             [borrower_lgd, corr_insured, 'Maturity_adj', "Years_to_maturity"]].apply(
        lambda s: func_risk_weight_corp(CMHC_pd, s[0], s[1], s[2], s[3]), axis=1)
    df.loc[cond_PMI, risk_weight_insured] = df.loc[:,
                                            [borrower_lgd, corr_insured, 'Maturity_adj', "Years_to_maturity"]].apply(
        lambda s: func_risk_weight_corp(CMHC_pd, s[0], s[1], s[2], s[3]), axis=1)

    # 1. CMHC insured
    df.loc[cond_CMHC, rwa] = df.loc[cond_CMHC, [risk_weight_insured, exposure]].apply(
        lambda s: RWA_res_mortgage(0, s[0], s[1], 1), axis=1)

    # 2. Sagen and CG
    df.loc[cond_PMI, rwa] = df.loc[
        cond_PMI, [risk_weight_uninsured, risk_weight_insured, exposure, deductible_ratio]].apply(
        lambda s: RWA_res_mortgage(s[0], s[1], s[2], s[3]), axis=1)

    # 3. Uninsured Loans
    df.loc[cond_UI, rwa] = df.loc[cond_UI, [risk_weight_uninsured, exposure]].apply(
        lambda s: RWA_res_mortgage(s[0], 0, s[1], 0), axis=1)
    return df

def airb_rwa_calc_2023_corp_v2(df, CMHC_pd, col_in=None,
                            col_out=None):
    if col_in is None:
        col_in = ['CalibratedPD', 'Final_LGD', 'EAD', 'insured_PMI_ratio', 'Advance_Amount']
    if col_out is None:
        col_out = ['corr_uninsured', 'corr_insured', 'risk_weight_uninsured', 'risk_weight_insured',
                   'rwa']
    import pandas as pd
    import numpy as np
    # import scipy.stats as st
    from myFunctions import func_risk_weight_retail, func_risk_weight_corp, RWA_res_mortgage, func_maturity_adj, \
        correlation_corp
    from sfr00_param_controls import correlation_residential_mortgages, correlation_residential_mortgages_rental, \
        dti_raw

    borrower_pd = col_in[0]
    borrower_lgd = col_in[1]
    exposure = col_in[2]
    deductible_ratio = col_in[3]
    origination_amount = col_in[4]

    corr_uninsured = col_out[0]
    corr_insured = col_out[1]
    risk_weight_uninsured = col_out[2]
    risk_weight_insured = col_out[3]
    rwa = col_out[4]

    cond_CMHC = (df.Insured_class.isin(['CMHC']))
    cond_PMI = (df.Insured_class.isin(["Sagen", "CG"])) & (df[exposure] > 0.1 * df[origination_amount])
    cond_UI = (df.Insured_class.isin(["Uninsured"])) | (
            (df.Insured_class.isin(["Sagen", "CG"])) & (df[exposure] <= 0.1 * df[origination_amount]))
    cond_rental = (df.Rental_Income == 'Y')

    df['Maturity_Date'] = pd.to_datetime(df['Maturity_Date'])
    dti = pd.to_datetime(dti_raw)
    df.loc[:, "Days_to_maturity"] = (df.Maturity_Date - dti).dt.days
    df.loc[:, "Years_to_maturity"] = df.loc[:, "Days_to_maturity"] / 365.25
    df['Years_to_maturity'] = np.where(df.Years_to_maturity <= 1, 1,
                                       np.where(df.Years_to_maturity >= 5, 5,
                                                df.Years_to_maturity))
    df.loc[cond_UI, "Maturity_adj"] = df.loc[:, [borrower_pd]].apply(lambda s: func_maturity_adj(s[0]), axis=1)
    df.loc[cond_CMHC | cond_PMI, "Maturity_adj"] = func_maturity_adj(CMHC_pd)

    df.loc[:, corr_uninsured] = np.where(cond_rental, correlation_residential_mortgages_rental,
                                         correlation_residential_mortgages)

    # correlation for CMHC and insured deductible for PMI ?
    df.loc[cond_CMHC | cond_PMI, corr_insured] = correlation_corp(CMHC_pd)
    # df.loc[cond_PMI, corr_insured] = df.loc[:, CMHC_pd].apply(lambda s: correlation_corp(s[0]), axis=1)

    ##### retail risk weight function for uninsured loans
    df[risk_weight_uninsured] = 0
    df.loc[cond_UI, risk_weight_uninsured] = df.loc[:, [borrower_pd, borrower_lgd, corr_uninsured]].apply(
        lambda s: func_risk_weight_retail(s[0], s[1], s[2]), axis=1)

    ####### re-write uninsred deductable for PMI using borrower PD and 100% LGD and retail risk weight function
    df.loc[cond_PMI, risk_weight_uninsured] = df.loc[:, [borrower_pd, corr_uninsured]].apply(
        lambda s: func_risk_weight_retail(s[0], 1, s[1]), axis=1)

    #### insured CMHC including PMI insured backstop use corporaterisk function
    df[risk_weight_insured] = 0

    df.loc[cond_CMHC, risk_weight_insured] = df.loc[:,
                                             [borrower_lgd, corr_insured, 'Maturity_adj', "Years_to_maturity"]].apply(
        lambda s: func_risk_weight_corp(CMHC_pd, s[0], s[1], s[2], s[3]), axis=1)
    df.loc[cond_PMI, risk_weight_insured] = df.loc[:,
                                            [borrower_lgd, corr_insured, 'Maturity_adj', "Years_to_maturity"]].apply(
        lambda s: func_risk_weight_corp(CMHC_pd, s[0], s[1], s[2], s[3]), axis=1)

    # 1. CMHC insured
    df.loc[cond_CMHC, rwa] = df.loc[cond_CMHC, [risk_weight_insured, exposure]].apply(
        lambda s: RWA_res_mortgage(0, s[0], s[1], 1), axis=1)

    # 2. Sagen and CG
    df.loc[cond_PMI, rwa] = df.loc[
        cond_PMI, [risk_weight_uninsured, risk_weight_insured, exposure, deductible_ratio]].apply(
        lambda s: RWA_res_mortgage(s[0], s[1], s[2], s[3]), axis=1)

    # 3. Uninsured Loans
    df.loc[cond_UI, rwa] = df.loc[cond_UI, [risk_weight_uninsured, exposure]].apply(
        lambda s: RWA_res_mortgage(s[0], 0, s[1], 0), axis=1)
    return df