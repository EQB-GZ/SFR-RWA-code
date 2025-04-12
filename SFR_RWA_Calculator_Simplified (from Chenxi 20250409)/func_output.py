def rwa_summary(df_in, group,col_pd):
    import pandas as pd

    def wavg(df, value, weight):
        v = df[value]
        w = df[weight]
        try:
            return (v * w).sum() / w.sum()
        except ZeroDivisionError:
            return v.mean()

    portfolio_summary = df_in.groupby([group]).agg({'Loan_Number': 'count',
                                                    'EAD': 'sum',
                                                    'RWA_AIRB': 'sum',
                                                    'RWA_standardized': 'sum',
                                                    'RWA_AIRB_2019':'sum',
                                                    'RWA_standardized_2019':'sum'
                                                    })
    pd_weighted_average = df_in.groupby([group]).apply(wavg, col_pd, 'EAD')
    lgd_weighted_average = df_in.groupby([group]).apply(wavg, 'Final_LGD', 'EAD')

    summary_wa = pd.DataFrame(data=dict(s1=pd_weighted_average, s2=lgd_weighted_average))
    summary_wa.columns = ['WeightedAvg_PD','WeightedAvg_LGD']

    # summary_tbl = pd.merge(portfolio_summary, pd_weighted_average, on=group)
    summary_tbl = pd.merge(portfolio_summary, summary_wa, on=group)

    return summary_tbl
