################################################################
####                                                        ####
####      CLTV Prediction with BG-NBD and Gamma-Gamma       ####
####                                                        ####
################################################################

# 1. Data Preparation
# 2. Expected Sales Forecast with BG-NBD Model
# 3. Expected Average Profit with the Gamma-Gamma Model
# 4. Calculation of CLTV with BG-NBD and Gamma-Gamma Model
# 5. Creation of Segments by CLTV
# 6. Functionalization of Work
# 7. Submitting Results to Database

# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
# Customer_Value = Average_Order_Value * Purchase_Frequency

################################################################
                      # Data Preperation #
################################################################
# Required Library and Functions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

# It can be used in case of error.

# pip install lifetimes
# pip install sqlalchemy
# pip  sqlalchemy
# pip install mysql-connector-python

################################################################
                 # Reading Data from Excel #
################################################################

df_ = pd.read_excel(r"C:\Users\hp\PycharmProjects\VBO\WEEK_03\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df = df_.copy()
df.head()

################################################################
               # Reading Data from Database #
################################################################
# Faster to pull data from database

from sqlalchemy import create_engine

# Credentials
creds = {
    "user" : "*******",
    "passwd" : "******",
    "host" : "******",
    "port" : ******,
    "db" : "*****"
}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))
# conn.close()

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)

retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)

retail_mysql_df.shape
retail_mysql_df.head()
retail_mysql_df.info()

################################################################
                   # Data Preprocessing #
################################################################

df = df[df["Country"] == "United Kingdom"]
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

df.describe().T

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

################################################################
         # Preparation of Lifetime Data Structure #
################################################################

# recency: Time since last purchase. Weekly.
# (It is used according to the analysis day in RFM and user-specific in CLTV.)

# T: Customer's age. Weekly.
# (How long before the analysis date the first purchase was made)

# frequency: Total number of repeat purchases (frequency>1)

# monetary_value: Average earnings per purchase

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.head()

# Edit the index
cltv_df.columns = cltv_df.columns.droplevel(0)

# Variable names
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df.head()

# Expressing monetary value as average earnings per purchase
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# Choosing those greater than monetary zero
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df.head()

# If a type error related to f is received;
# cltv_df["frequency"] = cltv_df["frequency"].astype(int)

# Expression of recency and T for BGNBD in weekly terms
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# frequency must be greater than 1.
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df.head()

################################################################
             # Installation of BG-NBD Model #
################################################################

from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# Who are the 10 customers we expect the most to purchase in a week?
################################################################
cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                cltv_df['frequency'],
                                                cltv_df['recency'],
                                                cltv_df['T'])

cltv_df.head()

# Who are the 10 customers we expect the most to purchase in a month?
################################################################
bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


cltv_df.sort_values("expected_purc_1_month", ascending=False).head(20)

# What is the Expected Number of Sales of the Whole Company in 1 Month?
################################################################

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

# What is the Expected Number of Sales of the Whole Company in 3 Months?
################################################################
bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

# Evaluation of Forecast Results
################################################################
import matplotlib.pyplot as plt
plot_period_transactions(bgf)
plt.show()

################################################################
        # Installation of GAMMA-GAMMA Model #
################################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values("expected_average_profit", ascending=False).head(20)
cltv_df.head()

################################################################
   # Calculation of CLTV with BG-NBD and Gamma-Gamma Model #
################################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 monthly
                                   freq="W",  # T: frequency
                                   discount_rate=0.01)

cltv.head()
cltv.shape

cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head()
cltv_final.sort_values(by="clv", ascending=False)[10:30]

#1-100 Transform
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(1, 100))
scaler.fit(cltv_final[["clv"]])
cltv_final["SCALED_CLTV"] = scaler.transform(cltv_final[["clv"]])
cltv_final.sort_values(by="clv", ascending=False)[10:30]
cltv_final.head()

# 1. Calculate 1-month and 12-month CLTV for 2010-2011 UK customers.
################################################################
cltv1 = ggf.customer_lifetime_value(bgf,
                                    cltv_df['frequency'],
                                    cltv_df['recency'],
                                    cltv_df['T'],
                                    cltv_df['monetary'],
                                    time=1,
                                    freq="W",
                                    discount_rate=0.01)

rfm_cltv1_final = cltv_df.merge(cltv1, on="Customer ID", how="left")
rfm_cltv1_final.sort_values(by="clv", ascending=False).head(10)

# 2. Analyze the top 10 people at 1 month CLTV and the 10 highest at 12 months. Is there a difference?
################################################################

rfm_cltv1_final.sort_values("clv", ascending=False).head(10)
rfm_cltv1_final.sort_values("clv", ascending=False).head(10)

# 3. For 2010-2011 UK customers, divide all your customers into 4 segments based on 6-month CLTV and add the group names to the dataset.
################################################################
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.head()

scaler = MinMaxScaler(feature_range=(1, 100))

scaler.fit(cltv_final[["clv"]])
cltv_final["SCALED_CLTV"] = scaler.transform(cltv_final[["clv"]])

cltv_final["cltv_segment"] = pd.qcut(cltv_final["SCALED_CLTV"], 4, labels=["D", "C", "B", "A"])
cltv_final["cltv_segment"].value_counts()
cltv_final.head()


cltv_final.groupby("cltv_segment")[["expected_purc_1_month", "expected_average_profit", "clv", "SCALED_CLTV"]].agg(
    {"count", "mean", "sum"})

# Segment A: Customers in this segment have the largest CLTV values.
# Parallel to this, the total expenditures, the number of transactions and the number of shopping are the highest.
# 391391.92521 was communicated to us as expected average profitability, while 643 transactions were made.
# Here, approximately 608 units of money per transaction are profitable.

################################################################
            # Submitting Results to Database #
################################################################

# Submit the final table from the candidates to your database.
# table in the form of the surname of the first name.
# The name "name" must be entered in the table content.

# Customer ID, recency, T, frequency, monetary, expected_purc_1_week,
# expected_purc_1_month, expected_average_profit
# clv, scaled_clv, segment

cltv_final = cltv_final.reset_index()
cltv_final["Customer ID"] = cltv_final["Customer ID"].astype(int)
cltv_final.head()

cltv_final.to_sql(name="Enes_Ozturk",con =conn,if_exists = 'replace',index = False)
pd.read_sql_query("show tables",conn)
conn.close()