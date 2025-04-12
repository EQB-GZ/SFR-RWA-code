###Functions###
def pd_estimator(values, coeff, intercept):
    ## Calculates the pd of drivers for a logistic regression.
    ## Inputs:
    # values: List of drivers, the order should match the order of coefficients
    # coeff: List of coefficients, the order should match the order of values
    # intercept: The intercept of the logistic regression
    ## Outputs:
    # The estimated PD as a float
    import numpy as np
    coeff1 = list(coeff)
    factor = sum([a * b for a, b in zip(values, coeff1)] + [intercept])
    pd = 1 / (1 + np.exp(-1 * factor))
    # return pd
    return round(pd, 4)  # for matching Business calculation rounding rules


def MRS_binning(df, MRS, pd_MRS, bin_MRS, label='pd', lower_inclusion=True):
    ## Mapping mapped values for pd/LGD.
    ## Inputs:
    # df: The dataframe including the "label"
    # label: The recorded continuous values
    # intercept: The intercept of the logistic regression
    ## Outputs:
    # The estimated PD as a float
    if lower_inclusion:
        for i in range(len(MRS) - 1):
            lb = MRS[i]
            ub = MRS[i + 1]
            cond = (df.loc[:, label] >= lb) & (df.loc[:, label] < ub)
            df.loc[cond, "MRS_bin" + label] = bin_MRS[i]
            df.loc[cond, "MRS_" + label] = pd_MRS[i]
    else:
        for i in range(len(MRS) - 1):
            lb = MRS[i]
            ub = MRS[i + 1]
            cond = (df.loc[:, label] > lb) & (df.loc[:, label] <= ub)
            df.loc[cond, "MRS_bin" + label] = bin_MRS[i]
            df.loc[cond, "MRS_" + label] = pd_MRS[i]
    return df


def woe_mapper(df, target, bounds, woe, woe_missing, label, discrete=False):
    df.loc[:, label] = woe_missing

    if discrete:

        for i in range(len(bounds)):
            cond = (df.loc[:, target] == bounds[i])
            df.loc[cond, label] = woe[i]
    else:
        for i in range(len(bounds) - 1):
            lb = bounds[i]
            ub = bounds[i + 1]
            cond = (df.loc[:, target] > lb) & (df.loc[:, target] <= ub)
            df.loc[cond, label] = woe[i]

    return df


def Renewal_Date_cleaner(s):
    import numpy as np
    try:
        return int('20' + str(s)[-2:] + str(s)[:2])
    except:
        return np.nan


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


def RWA_res_mortgage(capital_value, capital_value_insured, EAD_value, insured_ratio):
    # insured_ratio: Ratio of insured
    #     insured_RWA = round(capital_value_insured * 12.5 * EAD_value,0)
    #     uninsured_RWA = round(capital_value * 12.5 * EAD_value,0)
    insured_RWA = capital_value_insured * 12.5 * EAD_value
    uninsured_RWA = capital_value * 12.5 * EAD_value
    rwa_total = insured_ratio * insured_RWA + (1 - insured_ratio) * uninsured_RWA
    return rwa_total


def Standardized_RWA_res_mortgage(BCAR_risk_weight, EAD_value):
    rwa = (BCAR_risk_weight * EAD_value)
    return rwa


def correlation_corp(pd_value):
    import numpy as np
    #     corr =  round(0.12 * (1 - np.exp(-50 * pd_value)) / (1- np.exp(-50)) +
    #                       0.24 * (1 - (1- np.exp(-50 * pd_value))/(1 - np.exp(-50))),6) ##CAR 2023 5.3.1(i) 66.
    corr = 0.12 * (1 - np.exp(-50 * pd_value)) / (1 - np.exp(-50)) + 0.24 * (
            1 - (1 - np.exp(-50 * pd_value)) / (1 - np.exp(-50)))
    return corr


def maturity_adj(pd_value):
    import numpy as np
    adj = (0.11852 - 0.05478 * np.log(pd_value)) ** 2
    return adj  ##CAR 2023 5.3.1(i) 66.


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

def func_maturity_adj(pd_value):
    import numpy as np
    adj = (0.11852 - 0.05478 * np.log(pd_value)) ** 2
    return adj  ##CAR 2023 5.3.1(i) 66.
