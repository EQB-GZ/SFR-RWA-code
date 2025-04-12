import pandas as pd

YearTime = str(input("What is the year and month you are looking for (e.g. YYYYMM): "))
dict_month = {'01': 'Jan',
              '02': 'Feb',
              '03': 'Mar',
              '04': 'Apr',
              '05': 'May',
              '06': 'Jun',
              '07': 'Jul',
              '08': 'Aug',
              '09': 'Sep',
              '10': 'Oct',
              '11': 'Nov',
              '12': 'Dec'
              }
sql_yr = str('Y' + YearTime[0:4])
sql_month = str(YearTime[4:6])
sql_time = dict_month[sql_month]
Date = int(YearTime)
dti_raw = pd.Period(str(YearTime[0:4] + '-' + sql_month + '-01'), freq='M').end_time.date()
month_start = str(YearTime[0:4] + '-' + sql_month + '-01')

# sql_yr=str(input("What is the year you are looking for (e.g. 'Y2022'): " ))

# sql_time=str(input("What is the month you are looking for (e.g. 'Mar'): "))


# Date_input = input("Enter Date YYYYMM (int): ")
# Date = int(Date_input)

# sql_yr=str("Y"+Date[0:4])


# macro = [[202122, -0.007412, 6.846938],
#          [202203, -0.005778, 6.846938],
#          [202206, -0.005600, 5.528018],
#         [202209, 0.002603, 4.968029]]
# macro = pd.DataFrame(macro, columns=['date', 'pd_macro', 'lgd_macro'])
# PD_macro = macro.loc[macro.date == Date, 'pd_macro'].values[0]
# macro_LGD = macro.loc[macro.date == Date, 'lgd_macro'].values[0]
# This is based on the LGD MS
LGD_cure = 0.0072
# This is required for the DLGD calculations (according to LGD MS)
LR_LGD = 0.044540
# This is the LGD floor.
OSFI_DownturnLGD = 0.1
# This generic value is used to label values for TDS, GDS.
# Essentially the number should be small enough so that the
# actual values never reach that. Keep it static unless data
# requires its change.
missing_cons = -1000.0
pd_MRS = [0.0013, 0.0034, 0.0052, 0.0079, 0.0115, 0.0169, 0.0268, 0.1002]
LGD_MRS = [0.0509000000, 0.1408000000, 0.2040000000, 0.4007000000]

# CMHC_pd = 0.00001
# CMHC_pd_EY = 0.0001
pd_substitution = 0.00000001
correlation_residential_mortgages = 0.15
correlation_residential_mortgages_rental = 0.22

occupancy_ref = {"O": "Other",
                 "OO": "Owner Occupied",
                 "OOR": "Owner Occupied and Rental",
                 "RI": "Rental Income",
                 "SEC": "Secondary/Vacation",
                 "WLOO": "Work Live : Owner Occupied",
                 "WLOR": "Work Live :Owner Occupied and Rental",
                 "WLRI": "Work Live: Rental",
                 "A": "Applicant - No longer in Used",
                 "NEU": "No End Use"}

# Dictionary to map the occupancy status
rental_ref = {"OO": "Y",
              "OOR": "Y",
              "OOP": "Y",
              "WLOO": "Y",
              "WLOR": "Y",
              "O": "N",
              "RI": "N",
              "SEC": "N",
              "WLRI": "N",
              "A": "Y",
              "NEU": "N"}

rental_income_ref = {"OO": "N",  # Owner Occupied
                     "OOR": "Y",  # Owner Occupied and Rental (TBD)
                     "WLOO": "N",  # Work Live : Owner Occupied
                     "WLOR": "Y",  # Work Live :Owner Occupied and Rental (TBD)
                     "O": "Y",  # Other
                     "RI": "Y",  # Rental Income
                     "SEC": "Y",  # Secondary/Vacation
                     "WLRI": "Y",  # Work Live: Rental
                     "A": "N",  # Applicant - No longer in Used
                     "NEU": "N"  # No End Use
                     }

dwelling_ref = {'001': 'Single',
                '002': 'Semi-Detached',
                '003': 'Duplex',
                '004': 'Row',
                '005': 'Apartment',
                '006': 'Mobile',
                '008': 'Triplex',
                '009': 'Other',
                '010': 'Stacked',
                '017': 'Modular',
                '018': 'Fourplex',
                '019': 'Detached',
                '020': 'Condominium'}
