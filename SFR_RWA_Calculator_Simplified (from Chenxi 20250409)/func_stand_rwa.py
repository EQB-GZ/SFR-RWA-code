def stand_rwa_calc_2019(df, col_pd,col_insured, cols_in=None,
                        cols_out=None):
    if cols_out is None:
        cols_out = ['BCAR_weight_adj_2019', 'RWA_standardized_2019']
    if cols_in is None:
        cols_in = ['BCAR_Weighting', 'Current_LTV', 'EAD']
    import numpy as np
    from myFunctions import Standardized_RWA_res_mortgage
    LTV_label = cols_in[1]

    # df.loc[:,cols_out[0]] = df.loc[:,cols_in[0]]/100     #BCAR weighting filling from data
    # df.loc[:,"BCAR_weight_adj"] = df.loc[:,"BCAR_weight_mapped"]         #BCAR weighting filling from business
    # df.loc[:,"BCAR_weight_adj"] =np.nan

    cond_non_default = (df.loc[:, col_pd] != 1)
    cond_default = (df.loc[:, col_pd] == 1)
    cond_CMHC = (df.loc[:, col_insured].isin(["CMHC"]))
    # cond_PMI= (df.loc[:,'Insured_class'].isin(["CG","Sagen"])) #CAR2019 Chapter 3 paragrah 22, assume rating grade
    cond_CG = (df.loc[:, col_insured] == "CG")
    cond_Sagen = (df.loc[:, col_insured] == "Sagen")
    cond_uninsured = (df.loc[:, col_insured].isin(["Uninsured"]))

    rwa_conds = [
        cond_non_default & cond_CMHC,
        # cond_non_default & cond_PMI,
        cond_non_default & cond_CG,
        cond_non_default & cond_Sagen,
        cond_non_default & cond_uninsured & (df.loc[:, LTV_label] <= 80),
        cond_non_default & cond_uninsured & (df.loc[:, LTV_label] > 80),
        cond_default
    ]
    raw_bcar = [0, 0.02, 0.02, 0.35, 0.75, 1]
    df[cols_out[0]] = np.select(rwa_conds, raw_bcar, default=np.nan)
    # df[cols_out[0]] = np.where(df.BCAR_Weighting.isna(), df[cols_out[0]], df.BCAR_Weighting/100)
    df[cols_out[1]] = df.loc[:, [cols_out[0], cols_in[2]]].apply(lambda s: Standardized_RWA_res_mortgage(s[0], s[1]),
                                                                 axis=1)
    return df


def stand_rwa_calc_2023(df, col_pd,col_rental,col_insured, cols_in=None,
                        cols_out=None):
    if cols_out is None:
        cols_out = ['BCAR_weight_adj', 'RWA_standardized']
    if cols_in is None:
        cols_in = ['Current_LTV', 'EAD', 'insured_PMI_ratio','Advance_Amount']
    import numpy as np
    from myFunctions import Standardized_RWA_res_mortgage
    LTV_label = cols_in[0]

    # df.loc[:,"BCAR_weight_adj"] = df.loc[:,"BCAR_Weighting"]/100     #BCAR weighting filling from data
    # df.loc[:,"BCAR_weight_adj"] = df.loc[:,"BCAR_weight_mapped"]         #BCAR weighting filling from business
    # df.loc[:,"BCAR_weight_adj"] =np.nan

    cond_non_default = (df.loc[:, col_pd] != 1)
    # cond_default=(df.loc[:,"pd_test"] == 1)
    cond_rental = (df[col_rental] == "Y")
    cond_CMHC = (df.loc[:, col_insured].isin(["CMHC"]))
    # cond_PMI= (df.loc[:,'Insured_class'].isin(["CG","Sagen"])) #CAR 2023 Chapter 4 paragraph 61 cond_uninsured= (
    # df.loc[:,'Insured_class'].isin(["Uninsured"])) cond_PMI = (df.Insured_class.isin(["Sagen","CG"])) & (
    # df.Remaining_Principal_Post_CRM > 0.1*df.Advance_Amount) cond_uninsured = (df.Insured_class.isin([
    # "Uninsured"])) | ((df.Insured_class.isin(["Sagen","CG"])) & (df.Remaining_Principal_Post_CRM <=
    # 0.1*df.Advance_Amount)) cond_PMI = (df.Insured_class.isin(["Sagen","CG"])) & (df.EAD > 0.1*df.Advance_Amount)
    cond_CG = (df[col_insured] == "CG") & (df[cols_in[1]] > 0.1 * df[cols_in[3]])
    cond_Sagen = (df[col_insured] == "Sagen") & (df[cols_in[1]] > 0.1 * df[cols_in[3]])

    cond_uninsured = (df[col_insured].isin(["Uninsured"])) | (
            (df[col_insured].isin(["Sagen", "CG"])) & (df[cols_in[1]] <= 0.1 * df[cols_in[3]]))

    rwa_conds = [
        cond_non_default & cond_CMHC,
        # cond_non_default & cond_PMI,
        cond_non_default & cond_CG,
        cond_non_default & cond_Sagen,
        cond_non_default & ~cond_rental & cond_uninsured & (df.loc[:, LTV_label] <= 50),
        cond_non_default & ~cond_rental & cond_uninsured & ((50 < df.loc[:, LTV_label]) & (df.loc[:, LTV_label] <= 60)),
        cond_non_default & ~cond_rental & cond_uninsured & ((60 < df.loc[:, LTV_label]) & (df.loc[:, LTV_label] <= 70)),
        cond_non_default & ~cond_rental & cond_uninsured & ((70 < df.loc[:, LTV_label]) & (df.loc[:, LTV_label] <= 80)),
        cond_non_default & ~cond_rental & cond_uninsured & ((80 < df.loc[:, LTV_label]) & (df.loc[:, LTV_label] <= 90)),
        cond_non_default & ~cond_rental & cond_uninsured & (
                (90 < df.loc[:, LTV_label]) & (df.loc[:, LTV_label] <= 100)),
        cond_non_default & ~cond_rental & cond_uninsured & (df.loc[:, LTV_label] > 100),
        cond_non_default & cond_rental & cond_uninsured & (df.loc[:, LTV_label] <= 50),
        cond_non_default & cond_rental & cond_uninsured & ((50 < df.loc[:, LTV_label]) & (df.loc[:, LTV_label] <= 60)),
        cond_non_default & cond_rental & cond_uninsured & ((60 < df.loc[:, LTV_label]) & (df.loc[:, LTV_label] <= 70)),
        cond_non_default & cond_rental & cond_uninsured & ((70 < df.loc[:, LTV_label]) & (df.loc[:, LTV_label] <= 80)),
        cond_non_default & cond_rental & cond_uninsured & ((80 < df.loc[:, LTV_label]) & (df.loc[:, LTV_label] <= 90)),
        cond_non_default & cond_rental & cond_uninsured & ((90 < df.loc[:, LTV_label]) & (df.loc[:, LTV_label] <= 100)),
        cond_non_default & cond_rental & cond_uninsured & (df.loc[:, LTV_label] > 100),
        ~cond_non_default
    ]
    # raw_bcar=[0,0.02,
    raw_bcar = [0, np.nan, np.nan,
                0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70,
                0.30, 0.35, 0.45, 0.50, 0.60, 0.75, 1.05,
                1]
    df[cols_out[0]] = np.select(rwa_conds, raw_bcar, default=np.nan)
    # uninsured 2.2 * exposure of bank, risk weight for PMI deductible portion
    df.loc[cond_CG, cols_out[0]] = 0.2 * 2.2 * (1 - df.loc[:, cols_in[2]])
    df.loc[cond_Sagen, cols_out[0]] = 0.2 * 2.2 * (1 - df.loc[:, cols_in[2]])

    df[cols_out[1]] = df.loc[:, [cols_out[0], cols_in[1]]].apply(lambda s: Standardized_RWA_res_mortgage(s[0], s[1]),
                                                                 axis=1)
    return df
