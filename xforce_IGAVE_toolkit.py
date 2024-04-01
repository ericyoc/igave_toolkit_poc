#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
# X-Force IGAVE Toolkit:
#       Import of Existing Datasets
#       Generate a Fake Dataset
#       Application of ML Algorithms with R-squared and Accuracy Calculations
#       Visualization Demonstrations
#       Exploration of Potential Useful Models (e.g., Hype Cycle)
#
# Final version on August 9, 2021
# Free to reuse and adapt this pythone code as necessary. Simply provide a honorable mention of authorship.
#
# May need to contact Gartner about use and/or application of Hype Cycle since trademarked
# May allow use for educational purposes but not sure at this point
# https://www.gartner.com/en/about/policies/content-compliance
# https://trademarks.justia.com/862/24/hype-86224040.html
#
# Calculations from https://www.researchgate.net/publication/334328064
# Nigel Carr, University College London
# A Mathaematical Jutification of the gartner Hype Curve
# A Mathematical Formulation of the Emerging Risk Curve and
# Justificatoin for the Gartner Hype Cycle
#
# Loonshot Book reference made with Loonshot R&D Project Tracker used in this python code base
#
#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
# Attention! This is prototype code to demonstrate feasibility ONLY!
#            This python code is HARD CODED to the specific datasets used or generated.
#            You change the dataset you MUST change the python code in order to run.
#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
# Runs on Windows Decktop Computer
# No Internet/Web Connectivity is Required
# Must install python and assocated python libraries/modules in order to run this Python code base
# Must install python Dash that will run the Flask web service
#
# Using Dash based on Flask web service
# https://dash.plotly.com/introduction
# Dash is an open source library, released under the permissive MIT license.
# 
# Use web browswer
# so use http://127.0.0.1:8057
# or use http://localhost:8057
#
# Localhost = 127.0.0.1
# Port = 8057 and the port can be any port that you specify
#
# Scikit-learn is a Machine Learning (ML) library used for creating and training ML algorithms.
#
#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

#Assumes installed Python 3, matplotlib, pandas, numpy, dash, pandas, plotly
#Assumes installed pip or pip3
#Otherwise, do the following

#py -m pip install matplotlib
#py -m pip install numpy
#py -m pip install mplcursors
#py -m pip install dash
#py -m pip install pandas
#py -m pip install plotly
#py -m pip install plotly.graph_objs
#py -m pip install plotly.express
#py -m pip install plotly.sklearn.model_selection
#py -m pip install sklearn
#py -m pip install matplotlib
#py -m pip install pydbgen

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

# Import modules

# sys module
import sys

# dash module
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# numpy module
import numpy as np

# matplotlib module
import matplotlib.pyplot as plt
from matplotlib import pyplot

# pandas module
import pandas
from pandas.plotting import scatter_matrix
from pandas import DataFrame
import pandas as pd

# plotly module
import plotly
import plotly.graph_objs as go
import plotly.express as px

# sklearn module
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# pydbgen module
from pydbgen import pydbgen

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
# Generate Synthetic Data
# https://pydbgen.readthedocs.io/en/latest/
# https://faker.readthedocs.io/en/latest/index.html

# instanciate the faker
faker = pydbgen.pydb()

# generate the fake data and save the csv and excel files
# Columns include   Company_Name,         DUNS_Num,    CIK_Num,   Year_x, Funding_Type,      Funding_Amt, Funding_Source, HC_Status, HC_Maturity, HC_Section,  HC_Phase
# Example:          Ginkgo Bioworks,    827811626,  1,648,827,   2009,   Government Award,  $99,981,     NSF,            In_Labs,   Embryonic,   On_The_Rise, Innovation_Trigger

#def genName(size = 1):
    #return (np.array([faker.fake.unique.name() for i in range(size)]))        # return an array of the specified size of names

def genDunsNo(size = 1):
    my_duns_no_list = ['827811626']
    return (np.array([faker.fake.word(ext_word_list=my_duns_no_list) for i in range(size)])) # return an array of the specified size of DUNS number

def genCIK_No(size = 1):
    my_duns_no_list = ['1648827']
    return (np.array([faker.fake.word(ext_word_list=my_duns_no_list) for i in range(size)])) # return an array of the specified size of CIK number

def genCompany(size = 1):
    faker.fake.unique.clear()                                          # allow previous values generated to be returned again.
    return (np.array([faker.fake.unique.company() for i in range(size)]))     # return an array of the specified size of companies

def genDate(size = 1, m = 50, std = 20):
    #arr = np.random.normal(m, std, size).astype(int)                   # generate random integers of mean m and std spec
    #arr = np.clip(arr, 20, 90)                                         # clip respecting boundaries
    #arr = np.datetime64('today') - arr * 365                           # generate the arrays of past dates
    #return (arr)                                                       # return an array of the specified size of names
    return (np.array([faker.fake.year() for i in range(size)]))         # generate year only directly
    
def genFundingAmt(size = 1):
     return (np.array([faker.fake.unique.random_number(digits=6, fix_len=False) for i in range(size)]))     # return an array of fund amoutns not greater than six figures

def genFundingSrc(size = 1):
    my_funding_src_list = ['NSF', 'DHHS', 'DARPA', 'Venture Capital']
    return (np.array([faker.fake.word(ext_word_list=my_funding_src_list) for i in range(size)]))

def genFundingType(size = 1):
    my_funding_typelist = ['Government Award', 'Seed', 'Series A', 'Series B', 'Series C', 'Series D', 'Series E', 'Series F']
    return (np.array([faker.fake.word(ext_word_list=my_funding_typelist) for i in range(size)])) # return an array of the specified size of CIK number

# Attributes of dataset
# Create 1000 data elements
df_size = 1000
def genDataset(size = df_size):
    data = {'Company_Name': genCompany(size),
            'DUNS_Num': genDunsNo(size),
            'CIK_Num': genCIK_No(size),
            #'Name': genName(size),
            'Year_x': genDate(size),
            'Funding_Type': genFundingType(size),
            'Funding_Amt': genFundingAmt(size),
            'Funding_Source':genFundingSrc(size)}
    return (pd.DataFrame(data))

# Create dataframe
df = genDataset()

# Add Hype Cycle Columns to dataframe
df.insert(7, 'HC_Status' , 'none', True)
df.insert(8,'HC_Maturity', 'none', True)
df.insert(9,'HC_Section', 'none', True)
df.insert(10,'HC_Phase', 'none', True)

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
# Map to Hype Cycle based on Funding Type

# Map 'HC_Status' to 'Funding_Type'
for index in df.index:
    if df.loc[index, 'Funding_Type'] == 'Government Award':
            df.loc[index, 'HC_Status'] = 'In_Labs'
    elif df.loc[index, 'Funding_Type'] == 'Seed':
            df.loc[index, 'HC_Status'] = 'Pilots_and_Deployments'
    elif df.loc[index, 'Funding_Type'] == 'Series A':
            df.loc[index, 'HC_Status'] = 'Pilots_and_Deployments'
    elif df.loc[index, 'Funding_Type'] == 'Series B':
            df.loc[index, 'HC_Status'] = 'Pilots_and_Deployments'
    elif df.loc[index, 'Funding_Type'] == 'Series C':
            df.loc[index, 'HC_Status'] = 'Pilots_and_Deployments'
    elif df.loc[index, 'Funding_Type'] == 'Series D':
            df.loc[index, 'HC_Status'] = 'Pilots_and_Deployments'
    elif df.loc[index, 'Funding_Type'] == 'Series E':
            df.loc[index, 'HC_Status'] = 'Pilots_and_Deployments'
    elif df.loc[index, 'Funding_Type'] == 'Series F':
            df.loc[index, 'HC_Status'] = 'Pilots_and_Deployments'
    elif df.loc[index, 'Funding_Type'] == 'Growth' or df.loc[index, 'Funding_Type'] == 'IPO':
            df.loc[index, 'HC_Status'] = 'Evolving_Technology_Capability'
    elif df.loc[index, 'Funding_Type'] == 'Established':
            df.loc[index, 'HC_Status'] = 'Technology_Is_Proven'
    elif df.loc[index, 'Funding_Type'] == 'Expansion':
            df.loc[index, 'HC_Status'] = 'Technology_Is_Commoditized'
    elif df.loc[index, 'Funding_Type'] == 'Decline':
            df.loc[index, 'HC_Status'] = 'Still_functional_No_New_Development'
    elif df.loc[index, 'Funding_Type'] == 'Obsolete':
            df.loc[index, 'HC_Status'] = 'Maintenance_Only'  

# Map 'HC_Maturity' to 'Funding_Type'
for index in df.index:
    if df.loc[index, 'Funding_Type'] == 'Government Award':
            df.loc[index, 'HC_Maturity'] = 'Embryonic'
    elif df.loc[index, 'Funding_Type'] == 'Seed':
            df.loc[index, 'HC_Maturity'] = 'Emerging'
    elif df.loc[index, 'Funding_Type'] == 'Series A':
            df.loc[index, 'HC_Maturity'] = 'Emerging'
    elif df.loc[index, 'Funding_Type'] == 'Series B':
            df.loc[index, 'HC_Maturity'] = 'Emerging'
    elif df.loc[index, 'Funding_Type'] == 'Series C':
            df.loc[index, 'HC_Maturity'] = 'Emerging'
    elif df.loc[index, 'Funding_Type'] == 'Series D':
            df.loc[index, 'HC_Maturity'] = 'Emerging'
    elif df.loc[index, 'Funding_Type'] == 'Series E':
            df.loc[index, 'HC_Maturity'] = 'Emerging'
    elif df.loc[index, 'Funding_Type'] == 'Series F':
            df.loc[index, 'HC_Maturity'] = 'Emerging'
    elif df.loc[index, 'Funding_Type'] == 'Growth' or df.loc[index, 'Funding_Type'] == 'IPO':
            df.loc[index, 'HC_Maturity'] = 'Adolescent'
    elif df.loc[index, 'Funding_Type'] == 'Established':
            df.loc[index, 'HC_Maturity'] = 'Early_Mainstream'
    elif df.loc[index, 'Funding_Type'] == 'Expansion':
            df.loc[index, 'HC_Maturity'] = 'Mature_Mainstream'
    elif df.loc[index, 'Funding_Type'] == 'Decline':
            df.loc[index, 'HC_Maturity'] = 'Legacy'
    elif df.loc[index, 'Funding_Type'] == 'Exit':
            df.loc[index, 'HC_Maturity'] = 'Obsolete'

# Map 'HC_Section' to 'HC_Maturity'
for index in df.index:
    if df.loc[index, 'HC_Maturity'] == 'Embryonic':
            df.loc[index, 'HC_Section'] = 'On_The_Rise'
    elif df.loc[index, 'HC_Maturity'] == 'Emerging' and df.loc[index, 'Funding_Type'] == 'Seed':
            df.loc[index, 'HC_Section'] = 'At_The_Peak'
    elif df.loc[index, 'HC_Maturity'] == 'Emerging' and df.loc[index, 'Funding_Type'] == 'Series A':
            df.loc[index, 'HC_Section'] = 'At_The_Peak'
    elif df.loc[index, 'HC_Maturity'] == 'Emerging' and df.loc[index, 'Funding_Type'] == 'Series B':
            df.loc[index, 'HC_Section'] = 'Sliding_Into_Trough'
    elif df.loc[index, 'HC_Maturity'] == 'Emerging' and df.loc[index, 'Funding_Type'] == 'Series C':
            df.loc[index, 'HC_Section'] = 'Sliding_Into_Trough'
    elif df.loc[index, 'HC_Maturity'] == 'Emerging' and df.loc[index, 'Funding_Type'] == 'Series D':
            df.loc[index, 'HC_Section'] = 'Sliding_Into_Trough'
    elif df.loc[index, 'HC_Maturity'] == 'Emerging' and df.loc[index, 'Funding_Type'] == 'Series E':
            df.loc[index, 'HC_Section'] = 'Sliding_Into_Trough'
    elif df.loc[index, 'HC_Maturity'] == 'Emerging' and df.loc[index, 'Funding_Type'] == 'Series F':
            df.loc[index, 'HC_Section'] = 'Sliding_Into_Trough'
    elif df.loc[index, 'HC_Maturity'] == 'Adolescent' and df.loc[index, 'Funding_Type'] == 'Growth' or df.loc[index, 'Funding_Type'] == 'IPO':
            df.loc[index, 'HC_Section'] = 'Sliding_Into_Trough'
    elif df.loc[index, 'HC_Maturity'] == 'Adolescent' and df.loc[index, 'Funding_Type'] == 'Growth' or df.loc[index, 'Funding_Type'] == 'IPO':
            df.loc[index, 'HC_Section'] = 'Climbing_the_Slope'
    elif df.loc[index, 'HC_Maturity'] == 'Early_Mainstream' and df.loc[index, 'Funding_Type'] == 'Established':
            df.loc[index, 'HC_Section'] = 'Climbing_the_Slope'
    elif df.loc[index, 'HC_Maturity'] == 'Mature_Mainstream' and df.loc[index, 'Funding_Type'] == 'Expansion':
            df.loc[index, 'HC_Section'] = 'Entering_the_Plateau'
    elif df.loc[index, 'HC_Maturity'] == 'Legacy' and df.loc[index, 'Funding_Type'] == 'Decline':
            df.loc[index, 'HC_Section'] = 'Entering_the_Plateau'
    elif df.loc[index, 'HC_Maturity'] == 'Obsolete' and df.loc[index, 'Funding_Type'] == 'Exit':
            df.loc[index, 'HC_Section'] = 'Entering_the_Plateau'

# Map 'HC_Phase' to 'HC_Section'
for index in df.index:
    if df.loc[index, 'HC_Section'] == 'On_The_Rise':
            df.loc[index, 'HC_Phase'] = 'Innovation_Trigger'
    elif df.loc[index, 'HC_Section'] == 'At_The_Peak':
            df.loc[index, 'HC_Phase'] = 'Peak_of_Inflated_Expectations'
    elif df.loc[index, 'HC_Section'] == 'Sliding_Into_Trough':
            df.loc[index, 'HC_Phase'] = 'Trough_of_Disillusionment'
    elif df.loc[index, 'HC_Section'] == 'Climbing_the_Slope':
            df.loc[index, 'HC_Phase'] = 'Slope_of_Enlightenment'
    elif df.loc[index, 'HC_Section'] == 'Entering_the_Plateau':
            df.loc[index, 'HC_Phase'] = 'Plateau_of_Productivity'

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
            
# Save dataframe to csv file located where this python script is located
# Will right over the previous version if csv file exists
# Once csv file has been created it can be imported and used in this python code base
df_fake_company_file_name = 'fake_dataset.csv'
df.to_csv(df_fake_company_file_name, index = False)
#df.to_excel('fake_dataset.xlsx', index = False)

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

#port global
#port can be changed to any number but try and avoid well known ports (e.g., 443, 80)
set_port = 8057

#slider globals
max_value = 51 # HC max
max_value1 = 25 # MA max
max_value2 = 25 # MS max
max_value3 = 25 # MC max
max_value4 = 25 # MK max
max_value5 = 25 # ER max
max_value6 = 25 # PM max
max_value7 = 2100 # FM max

#strings globals
hc_t1 = "Innovation Trigger: The trigger occurs before market awareness and the increase of hype represents the forces dirving actual later market awareness."
hc_t2 = "Peak of Inflation: The peak occurs when actual maarket awarness of the innovation starts to increase in response to the forces driving the awareness."
hc_t3 = "Trough of Disillusionment: The trough reaches its lowest level when the spread of knowledge about how to deploy the innovation catches up with the level of market awareness."
hc_t4 = "Slope of Enlightenment: The slop of enlightenment is drive by knowledge of how to deploy the innovation to achieve benefits."
hc_t5 = "Plateau of Productivity: The plateau of productivity is the asymptote represented by the sustainable level of use the innovation driven by deployment knowledge."
ms_t1 = "Innovators: These individuals adopt new technology or ideas simply because they are new. Innovators tend to take risks more readily and are the most venturesome."
ms_t2 = "Early Adopters: This group tends to create opinions, which propel trends. They are not unlike innovators in how quickly they take on new technologies and ideas but are more concerned about their reputation as being ahead of the curve."
ms_t3 = "Early Majority: If an idea or other innovation enters this group, it tends to be widely adopted before long. This group makes decisions based on utility and practical benefits over coolness."
ms_t4 = "Late Majority: The late majority shares some traits with the early majority but is more cautious before committing, needing more hand-holding as they adopt."
ms_t5 = "Laggards: This group is slow to adapt to new ideas or technology. They tend to adopt only when they are forced to or because everyone else has already."


# x and y axis labels
default_x_label = 'The year when the weapon appeared'
default_y_label = 'Log of Muzzle Kinetic Energy (J)'

bio_firm_x_label = "The year of funding"
bio_firm_y_label = "The amount of funding"

g_x_label = ""
g_y_label = ""

# slider max and min
g_slider_max = 1000
g_slider_min = 2100

# csv file global
cur_file_name = ""
default_file_name = "infratry_small_arms_data.csv"
df_future_map = pd.read_csv(default_file_name)  # default dataset for ML models
df_model_score = pd.read_csv(default_file_name) # default dataset for ML Model R-Squared and Accuracy calculations
df_real_company_file_name = "synth_bio_data.csv"
df_synth_bio = pd.read_csv(df_real_company_file_name) # default dataset with real data from golden.com web site
g_imported = False

# Other dataset sources:
# 
# Tradespace (http://www.atsv.psu.edu/)
# TRAC dataset (https://www.trac.army.mil/)
# LinkedIn (https://www.linkedin.com/help/linkedin/answer/2836/accessing-linkedin-apis?lang=en)
# Crunchbase (https://www.crunchbase.com/discover/organization.companies) 
# PitchBook (https://pitchbook.com/data)
# Golden (https://golden.com/list-of-synthetic-biology-companies/)
# OECD (https://www.oecd.org/unitedstates/) 
# Kaggle: 
#   https://www.kaggle.com/search?q=crunchbase+in%3Adatasets
#   https://www.kaggle.com/mirbektoktogaraev/should-this-loan-be-approved-or-denied 
#   https://www.kaggle.com/ajitpasayat/startup-success-prediction-dataset 
#   https://www.kaggle.com/iwanenko/unicorn-nest-startup-fundraising-dataset 
#   Army Research Lab (ARL) Specific
#
# models global
# use Scikit-learn to split and preprocess our data and train various regression models.
# Scikit-learn is a Machine Learning (ML) library that offers various tools for creating and training ML algorithms.
# Works with libraries such as NumPy and Pandas

# 8 different ML algorithms:
#   Linear Regression
#   Decision Tree Regression
#   K-Nearest Neighbors Regression
#   Logistic Regression
#   Linear Discriminant Analysis
#   Decision Tree Regression with AdaBoost 
#   Gaussian Naive Bayes Classifier
#   Linear Support Vector Machines

models = {'Linear Regression': sklearn.linear_model.LinearRegression,
          'Decision Tree Regression': sklearn.tree.DecisionTreeRegressor,
          'K-Nearest Neighbors Regression': sklearn.neighbors.KNeighborsRegressor,
          'Logistic Regression': sklearn.linear_model.LogisticRegression,
          'Linear Discriminant Analysis':sklearn.discriminant_analysis.LinearDiscriminantAnalysis,
          'Decision Tree Regression with AdaBoost':sklearn.ensemble.AdaBoostRegressor,
          'Gaussian Naive Bayes Classifier': sklearn.naive_bayes.GaussianNB,
          'Linear Support Vector Regression':LinearSVR,
         }

# set sample amount for fake dataset for ML models (see above models)
# currently set for df_size = 1000
sample_amount = 5000

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

# Hype Curve Dataset Generation
def hype_curve_formula(x):
    
    B = 100
    C = 25
    b = 0.8
    a = 0.1
    x0 = 10
    x2 = 5

    y = -(((B/(1+np.exp(-b*(x-x0-x2))))) -((B/(1+np.exp(-b*(x-x0))))) - ((C/(1+np.exp(-a*(x-x0-x2))))))
    return y

x_list = []
y_list = []

for i in range(1, max_value):
    x_list.append(i)

for i in range(1, max_value):
    y = hype_curve_formula(i)
    y_list.append(y)

df = pd.DataFrame(
    {'x': x_list,
     'y': y_list
    })

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

# Market Adoption Dataset Generation
def market_adoption_curve_formula(x):
    # Market Adoption Curve
    # market_adoption(x) = A*e^(-a*(x-x0))

    A = 50
    a = 0.1
    x0 = 10
    
    y1 = A*np.exp(-a*(x-x0) ** 2)
    return y1

x_list1 = []
y_list1 = []

for i in range(1, max_value1):
    x_list1.append(i)

for i in range(1, max_value1):
    y1 = market_adoption_curve_formula(i)
    y_list1.append(y1)

df_market_adoption = pd.DataFrame(
    {'x1': x_list1,
     'y1': y_list1
    })

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

# Market Share Dataset Generation
def market_share_curve_formula(x):
    # Market Share Curve = Market Awareness
    # market_share(x) = market_awareness = B / (1 + e^(-b*(x-x0)))

    B = 100
    b = 0.8
    x0 = 10
    
    y2 = (B/(1+np.exp(-b*(x-x0))))
    return y2

x_list2 = []
y_list2 = []

for i in range(1, max_value2):
    x_list2.append(i)

for i in range(1, max_value2):
    y2 = market_share_curve_formula(i)
    y_list2.append(y2)

df_market_share = pd.DataFrame(
    {'x2': x_list2,
     'y2': y_list2
    })

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

# Market Action Dataset Generation
def market_action_curve_formula(x):
    # Market Action Curve
    # market_action(x) = B / (1 + e^(-b*(x-x0-x1)))

    B = 100
    b = 0.8
    x0 = 10
    x1 = 8

    y3 = (B/(1+np.exp(-b*(x-x0-x1))))
    return y3

x_list3 = []
y_list3 = []

for i in range(1, max_value3):
    x_list3.append(i)

for i in range(1, max_value3):
    y3 = market_action_curve_formula(i)
    y_list3.append(y3)

df_market_action = pd.DataFrame(
    {'x3': x_list3,
     'y3': y_list3
    })

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

# Market Knowledge Dataset Generation
def market_knowledge_curve_formula(x):
    # Market Knowledge Curve
    # market_knowledge(x) = C / (1 + e^(-a*(x-x0-x2)))

    C = 25
    a = 0.1
    x0 = 10
    x2 = 5
    
    y4 = (C/(1+np.exp(-a*(x-x0-x2))))
    return y4

x_list4 = []
y_list4 = []

for i in range(1, max_value4):
    x_list4.append(i)

for i in range(1, max_value4):
    y4 = market_knowledge_curve_formula(i)
    y_list4.append(y4)

df_market_knowledge = pd.DataFrame(
    {'x4': x_list4,
     'y4': y_list4
    })

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

# Emerging Risk Dataset Generation
def emerging_risk_curve_formula(x):
    # Emerging Risk Curve
    # emerging_risk(x) = market_awareness(x) - market_knowledge(x)
    # Essentially 𝑥1 could be any number with 0<𝑥1<𝑥0.
    # Essentially 𝑥2 could be any number with 0<𝑥2<𝑥1<𝑥0.
    # market_share(x) = market_awarness(x)
    # emerging_risk(x) = market_share(x) - market_knowledge(x)
    # emerging_risk(x) = B / (1 + e^(-b*(x-x0))) - [C / (1 + e^(-a*(x-x0-x2)))]

    B = 100
    C = 25
    b = 0.8
    a = 0.1
    x0 = 10
    x2 = 5
    
    y5 = (B/(1+np.exp(-b*(x-x0)))) - ((C/(1+np.exp(-a*(x-x0-x2)))))
    return y5

x_list5 = []
y_list5 = []

for i in range(1, max_value5):
    x_list5.append(i)

for i in range(1, max_value5):
    y5 = emerging_risk_curve_formula(i)
    y_list5.append(y5)

df_emerging_risk = pd.DataFrame(
    {'x5': x_list5,
     'y5': y_list5
    })

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

# Innovation Persepction Map Dataset Generation
def innovation_perception_map_formula(x):

    #y6 = (x ** 2)
    
    y6 = np.cos(x)
    
    return y6

x_list6 = []
y_list6 = []

for i in range(1, max_value6):
    x_list6.append(i)

for i in range(1, max_value6):
    y6 = innovation_perception_map_formula(i)
    y_list6.append(y6)

df_perception_map = pd.DataFrame(
    {'x6': x_list6,
     'y6': y_list6
    })

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

# Update project location along line or curve based on slider
def track_along_curve_with_slider(cur_pos_x, cur_saved_y_pos, cur_x_range, cur_y_range):
    x_value = cur_pos_x
    saved_y_pos = cur_saved_y_pos
    y_value = saved_y_pos
    
    # Create list of x values from prediction
    x_list = []
    for item in cur_x_range:
        x_list.append(int(item))

    # Create list of y values from prediction
    y_list = []
    for item in cur_y_range:
        y_list.append(int(item))

    # Combine x and y values into df
    # query df for specific slider x value
    # generate the associated y value for the x value from slider
    # save the y value as last know y value
    try:
        xy_df = pd.DataFrame({'x': x_list, 'y' :y_list})
        my_y_value = xy_df.query("x ==["+str(x_value)+"]")
        y_value = my_y_value.y.values[0]
        saved_y_pos = y_value

    except IndexError:
        y_value = saved_y_pos
        pass
    
    return y_value, saved_y_pos

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

def calculate_r_squared_or_accuracy(visual_selected, model_selected, filename):

    if visual_selected == 'FM':

        #For regressions we calculate R-squared
        #R-squared score of 60% reveals that 60% of the data fits the regression model
        #Higher the R-squeared the better. Acceptable is 70% but want over 90%

        if g_imported == True:
            if default_file_name == filename:
                x_value = df_future_map.year.values.reshape(-1, 1)
                y_value = df_future_map.v.values.reshape(-1, 1)
                
            elif df_real_company_file_name == filename:
                df_cur_dataframe = df_imported.copy(deep=True)
                x_value = df_cur_dataframe.Year_x.values.reshape(-1, 1)
                y_value = df_cur_dataframe.Funding_Amt.values.reshape(-1, 1)
                
            elif df_fake_company_file_name == filename:
                df_cur_dataframe = df_imported.copy(deep=True)
                x_value = df_cur_dataframe.Year_x.values.reshape(-1, 1)
                y_value = df_cur_dataframe.Funding_Amt.values.reshape(-1, 1)
                
        elif g_imported == False:
            x_value = df_future_map.year.values.reshape(-1, 1)
            y_value = df_future_map.v.values.reshape(-1, 1)
            
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(x_value, y_value, test_size=0.3, random_state=0)

        if model_selected == 'Linear Regression':
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            
            model = LinearRegression()

        elif model_selected == 'Decision Tree Regression':

            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            
            model = DecisionTreeRegressor()

        elif model_selected == 'K-Nearest Neighbors Regression':

            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            
            model = KNeighborsRegressor()

        elif model_selected == 'Logistic Regression':

            model = LogisticRegression()

        elif model_selected == 'Linear Discriminant Analysis':
            
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            
            model = LinearDiscriminantAnalysis(n_components=1)

        elif model_selected == 'Decision Tree Regression with AdaBoost':

            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)

            model = AdaBoostRegressor()

        elif model_selected == 'Gaussian Naive Bayes Classifier':

            model = GaussianNB()

            # Fit Model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            
            model_prec = metrics.accuracy_score(y_test, y_pred)
            model_prec = np.round(model_prec, 2)
            model_prec = model_prec * 100

            result = str(model_prec)
            final_result = model_selected + " model accuracy is calculated to be " + result + "%"

            return final_result

        elif model_selected == 'Linear Support Vector Regression':

            model = LinearSVR()
            
        # Fit Model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        model_prec = np.absolute(metrics.r2_score(y_test, y_pred))
        model_prec = np.round(model_prec, 2)
        model_prec = model_prec * 100

        result = str(model_prec)
        final_result = model_selected + " model R-squared is calculated to be " + result + "%"

    return final_result 

# Setup Dash
app = dash.Dash(__name__)

# Setup Dash Layout
app.layout = html.Div([
    html.Div([
        html.H2("X-Force IGAVE Toolkit"),
        html.H3("Select Visualization:"), 
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': 'Future Map', 'value': 'FM'},
                {'label': 'Hype Cycle - Synthetic Biology', 'value': 'SB'},
                {'label': 'Hype Cycle', 'value': 'HC'},
                {'label': 'Market Adoption', 'value': 'MA'},
                {'label': 'Market Share', 'value': 'MS'},
                {'label': 'Market Action', 'value': 'MC'},
                {'label': 'Market Knowledge', 'value': 'MK'},
                {'label': 'Emerging Risk', 'value': 'ER'},
                {'label': 'Innovation Perception Map', 'value': 'PM'}
            ],
            value='FM', style={"margin-top": "0px", 'width': '50%', 'display': 'inline-block'},
        )]),
    
    html.Div([
        html.H3("Select Model:"),
        dcc.Dropdown(
            id='model-name',
            options=[{'label': x, 'value': x} 
                 for x in models
            ],
            value='Linear Regression',
            clearable=False,
            style={"margin-top": "0px", 'width': '50%', 'display': 'inline-block'},
         )]),

    html.Div([
        html.H3("Select Dataset to Load into Model:"),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File')
            ]),
            style={
                'width': '40%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
        html.Div(id='output-data-upload'),
    ]),
    
    html.Div([
        dcc.Graph(id="graph"
        )],style={"margin-top": "0px","margin-left": "0px",'width': '85%', 'display': 'inline-block'},
        ),
        html.A(html.Button('Refresh Page'),href='/'),
    
    html.Div([ 
        html.H3("Time:"), 
        dcc.Slider(
            id='slider', 
             marks={
                    2: {'label': 'Innovation Trigger', 'style': {'color': '#77b0b1', 'fontSize':10}},
                    13: {'label': 'Peak of Inflation Expectations', 'style': {'color': '#77b0b1','fontSize':10}},
                    20: {'label': 'Trough of Disillusionment', 'style': {'color': '#77b0b1','fontSize':10}},
                    30: {'label': 'Slope of Enlightenment', 'style': {'color': '#77b0b1','fontSize':10}},
                    45: {'label': 'Plateau of Productivity','style': {'color': '#77b0b1', 'fontSize':10,}}},

             value= 2,
             step = 1,
             updatemode='drag',
        ),
        html.Div(id='output-container-range-slider'),
        html.Div(id='output-model_accurracy', style={'display': 'block'})
        ],style={"margin-top": "0px","margin-left": "100px", 'width': '70%', 'display': 'inline-block'},id = 'slider_section',),
    html.Div([
        html.Blockquote(hc_t1),
        html.Blockquote(hc_t2),
        html.Blockquote(hc_t3),
        html.Blockquote(hc_t4),
        html.Blockquote(hc_t5)
        ], style={'color': 'grey', 'fontSize': 12},id='my_blockquotes'),
],style={"margin-top": "10px", "margin-left": "20px", 'width': '75%', 'float': 'left', 'display': 'inline-block'},id = 'phase_descriptions',)


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
# Setup Callbacks

@app.callback(
    dash.dependencies.Output('output-container-range-slider', 'children'),
    [dash.dependencies.Input('slider', 'value')])

def update_output(value):
    return 'Slider: Year "{}".'.format(value)

@app.callback(
    Output('output-model_accurracy', 'children'),
    [Input("dropdown", "value"),
     Input('model-name', 'value'),
     Input('upload-data', 'filename')])

def update_output(visual_selected, model_selected, filename):

    if visual_selected == 'FM':
        final_result = calculate_r_squared_or_accuracy(visual_selected, model_selected, filename)
    else:
        final_result = "No Model Available to Calcualte R-Squared or Accuracy"

    return final_result


@app.callback(Output('model-name', 'options'),
              Input('dropdown', 'value'))

def set_model_options(selected_visualization):
    if selected_visualization == 'HC':
        return [{'label': 'None Available', 'value': 'N'}]
    elif selected_visualization == 'SB':
        return [{'label': 'None Available', 'value': 'N'}]
    elif selected_visualization == 'MA':
        return [{'label': 'None Available', 'value': 'N'}]
    elif selected_visualization == 'MS':
        return [{'label': 'None Available', 'value': 'N'}]
    elif selected_visualization == 'MC':
        return [{'label': 'None Available', 'value': 'N'}]
    elif selected_visualization == 'MK':
        return [{'label': 'None Available', 'value': 'N'}]
    elif selected_visualization == 'ER':
        return [{'label': 'None Available', 'value': 'N'}]
    elif selected_visualization == 'PM':
        return [{'label': 'None Available', 'value': 'N'}]
    elif selected_visualization == 'FM':
        return [{'label': x, 'value': x} 
                 for x in models]
    
#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
@app.callback([Output('slider', 'marks'),
               Output('slider', 'min'),
               Output('slider', 'max')],
               [Input('dropdown', 'value'),
               Input('upload-data', 'filename')])

def update_slider_markers(my_drop_down_value, filename):

        if my_drop_down_value == 'HC':
                 marks = {
                    2: {'label': 'Innovation Trigger', 'style': {'color': '#77b0b1', 'fontSize':10}},
                    13: {'label': 'Peak of Inflation Expectations', 'style': {'color': '#77b0b1','fontSize':10}},
                    20: {'label': 'Trough of Disillusionment', 'style': {'color': '#77b0b1','fontSize':10}},
                    30: {'label': 'Slope of Enlightenment', 'style': {'color': '#77b0b1','fontSize':10}},
                    45: {'label': 'Plateau of Productivity','style': {'color': '#77b0b1', 'fontSize':10,}}
                }
                 min=df['x'].min()
                 max=df['x'].max()
                 
        elif my_drop_down_value == 'SB':
                 marks = {
                    2: {'label': 'Innovation Trigger', 'style': {'color': '#77b0b1', 'fontSize':10}},
                    13: {'label': 'Peak of Inflation Expectations', 'style': {'color': '#77b0b1','fontSize':10}},
                    20: {'label': 'Trough of Disillusionment', 'style': {'color': '#77b0b1','fontSize':10}},
                    30: {'label': 'Slope of Enlightenment', 'style': {'color': '#77b0b1','fontSize':10}},
                    45: {'label': 'Plateau of Productivity','style': {'color': '#77b0b1', 'fontSize':10,}}
                }
                 min=df['x'].min()
                 max=df['x'].max()
        
        elif my_drop_down_value == 'MA':
                marks = {
                    2: {'label': 'Innovators (2.5%)', 'style': {'color': '#77b0b1', 'fontSize':10}},
                    6: {'label': 'Early Adopters (13.5%)', 'style': {'color': '#77b0b1','fontSize':10}},
                    9: {'label': 'Early Majority (34%)', 'style': {'color': '#77b0b1','fontSize':10}},
                    13: {'label': 'Late Marjority (34%)', 'style': {'color': '#77b0b1','fontSize':10}},
                    20: {'label': 'Laggards (16%)','style': {'color': '#77b0b1', 'fontSize':10,}}
                }
                min=df_market_adoption['x1'].min()
                max=df_market_adoption['x1'].max()
                
        elif my_drop_down_value == 'MS':
                marks = {
                    2: {'label': '0% Market Share', 'style': {'color': '#77b0b1', 'fontSize':10}},
                    6: {'label': '12.5% Market Share', 'style': {'color': '#77b0b1','fontSize':10}},
                    9: {'label': '25% Market Share', 'style': {'color': '#77b0b1','fontSize':10}},
                    13: {'label': '50% Market Share)', 'style': {'color': '#77b0b1','fontSize':10}},
                    20: {'label': '100% Market Share','style': {'color': '#77b0b1', 'fontSize':10,}}
                }
                min=df_market_share['x2'].min()
                max=df_market_share['x2'].max()

        elif my_drop_down_value == 'MC':
                marks = {
                    2: {'label': '0% Market Share', 'style': {'color': '#77b0b1', 'fontSize':10}},
                    15: {'label': '12.5% Market Share', 'style': {'color': '#77b0b1','fontSize':10}},
                    18: {'label': '25% Market Share', 'style': {'color': '#77b0b1','fontSize':10}},
                    21: {'label': '50% Market Share)', 'style': {'color': '#77b0b1','fontSize':10}},
                    24: {'label': '100% Market Share','style': {'color': '#77b0b1', 'fontSize':10,}}
                }
                min=df_market_action['x3'].min()
                max=df_market_action['x3'].max()

        elif my_drop_down_value == 'MS':
                marks = {
                    2: {'label': '0% Market Share', 'style': {'color': '#77b0b1', 'fontSize':10}},
                    24: {'label': '100% Market Share','style': {'color': '#77b0b1', 'fontSize':10,}}
                }
                min=df_market_knowledge['x4'].min()
                max=df_market_knowledge['x4'].max()

        elif my_drop_down_value == 'MK':
                marks = {
                    0: {'label': '', 'style': {'color': '#77b0b1', 'fontSize':10}},
                    25: {'label': '', 'style': {'color': '#77b0b1', 'fontSize':10,}}
                }
                min=df_market_knowledge['x4'].min()
                max=df_market_knowledge['x4'].max()

        elif my_drop_down_value == 'ER':
                marks = {
                    0: {'label': '', 'style': {'color': '#77b0b1', 'fontSize':10}},
                    25: {'label': '', 'style': {'color': '#77b0b1', 'fontSize':10,}}
                }
                min=df_emerging_risk['x5'].min()
                max=df_emerging_risk['x5'].max()

        elif my_drop_down_value == 'PM':
                marks = {
                    0: {'label': '', 'style': {'color': '#77b0b1', 'fontSize':10}},
                    25: {'label': '', 'style': {'color': '#77b0b1', 'fontSize':10,}}
                }
                min=df_perception_map['x6'].min()
                max=df_perception_map['x6'].max()

        elif my_drop_down_value == 'FM':

                min=g_slider_max
                max=g_slider_min
                marks = {
                    int(min): {'label': '', 'style': {'color': '#77b0b1', 'fontSize':10}},
                    int(max): {'label': '', 'style': {'color': '#77b0b1', 'fontSize':10,}}
                }
    
        return marks, min, max

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    
@app.callback(Output('my_blockquotes', 'children'),
              Input('dropdown', 'value'))

def update_blockquotes(my_drop_down_value):

        if my_drop_down_value == 'HC':
            return [
                html.Blockquote(hc_t1),
                html.Blockquote(hc_t2),
                html.Blockquote(hc_t3),
                html.Blockquote(hc_t4),
                html.Blockquote(hc_t5)
            ]
        
        elif my_drop_down_value == 'MA':
            return [
                html.Blockquote(ms_t1),
                html.Blockquote(ms_t2),
                html.Blockquote(ms_t3),
                html.Blockquote(ms_t4),
                html.Blockquote(ms_t5)
            ]

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

@app.callback(
     Output("graph", "figure"),
     [Input("slider", "value"),
     Input("dropdown", "value"),
     Input('model-name', 'value'),
     Input('graph', "hoverData"),
     Input('upload-data', 'filename')])
    
# Display graph using app callback since using slider capability
def graph(pos_x, dropdown_selection, model_selection, get_hover_data, filename):

    df_cur_dataframe = pd.DataFrame() # empty dataframe
    df_imported = pd.DataFrame() # empty dataframe
    
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df_imported = pd.read_csv(filename)
            df_cur_dataframe = df_imported.copy(deep=True)
            cur_file_name = filename
            g_imported = True
            #print(filename)
            
    except Exception as e:
        #print(e)
        df_cur_dataframe = df_future_map.copy(deep=True)
        g_imported = False

    
    if dropdown_selection == "HC":
        'You have selected "{}"'.format(dropdown_selection)
        hover_name = "Hype Cycle"

        fig = px.line(df, x='x', y='y', title = "Loonshot R&D Project:")
        #fig.add_vline(
            #x=pos_x, line_width=3, line_dash="dash", 
           # line_color="green")
        fig.add_annotation(x=pos_x, y=hype_curve_formula(pos_x),
                text="Loonshot R&D Project",
                showarrow=True,
                arrowhead=3)
        fig.add_annotation(x=2, y=10,
                text="Innovation Trigger",
                showarrow=False,
                arrowhead=0)
        fig.add_annotation(x=15, y=90,
                text="Peak of Inflation Expectations",
                showarrow=False,
                arrowhead=0)
        fig.add_annotation(x=20, y=10,
                text="Trough of Disillusionment",
                showarrow=False,
                arrowhead=0)
        fig.add_annotation(x=30, y=12,
                text="Slope of Enlightenment",
                showarrow=False,
                arrowhead=0)
        fig.add_annotation(x=45, y=17,
                text="Plateau of Productivity",
                showarrow=False,
                arrowhead=0)
    
        fig.update_xaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(title='Market Hype or Expectations [%]') 
        fig.update_xaxes(title='Time') 
        fig.update_traces(hovertext=hover_name, hovertemplate=f'<b>{hover_name}<b>') # provide hover that names curve

    elif dropdown_selection == "SB":
        
        'You have selected "{}"'.format(dropdown_selection)
        hover_name = "Hype Cycle - Synthetic Biology"

        # load dataset items from the company dataset - file called synth_bio_data
        company_name = df_synth_bio["Company_Name"].values[0]
        duns_number = df_synth_bio["DUNS_Num"].values[0]
        cik_number = df_synth_bio["CIK_Num"].values[0]

        # original hype curve plot from formula => x is 1,2,3... and y = f(x) using formula
        fig = px.line(df, x='x', y='y', title = "Synthetic Biology Company: " + company_name + ", DUNS No.: " +
                      str(duns_number) + ", CIK No.: " + str(cik_number))
        #fig.add_vline(
            #x=pos_x, line_width=3, line_dash="dash", 
           # line_color="green")
           
        fig.add_annotation(x=pos_x, y=hype_curve_formula(pos_x),
                text=company_name,
                showarrow=True,
                arrowhead=3)
        
        # use loaded dataset to plot on the hype cycle
        # add trace plot here and adjust to Hype Cycle phase locations on curve
        get_x1 = df_synth_bio['year_x'].to_numpy(np.int)
        for i in range(0,len(get_x1)):
            get_x1[i] = int(get_x1[i])
            get_x1[i] = get_x1[i] % 100

        x1 = get_x1
        for i in range(0,len(x1)):
            if df_synth_bio["HC_Phase"].values[i] == 'Innovation_Trigger':
                x1[i] = x1[i] - 8   # adjust x input to fall within specific hype cycle phase location on curve
            elif df_synth_bio["HC_Phase"].values[i] == 'Peak_of_Inflated_Expectations':
                x1[i] = x1[i] - 2   # adjust x input to fall within specific hype cycle phase location on curve
            elif df_synth_bio["HC_Phase"].values[i] == 'Trough_of_Disillusionment':
                x1[i] = x1[i] + 2   # adjust x input to fall within specific hype cycle phase location on curve

        # calculate y based on x from dataset
        y1 = [None] * len(x1);  
        for i in range(0,len(x1)):
            y1[i] = hype_curve_formula(x1[i])  # calculate y based on x ajustment and using hype curve fomula
            #y1[i]= x1[i]

        # plot the points on the hype cycle curve from the dataset
        fig.add_trace(go.Scatter(x = x1, y=y1,
            mode="markers+text",
            name="Actual Tracking",
            #text=["P 1", "P 2", "P 3", "P 4", "P 5", "P 6", "P 7", "P 8", "P 9", "P 10", "P 11", "P 12", "P 13", "P 14", "P 15"],
            textposition="bottom center"))

        fig.add_annotation(x=2, y=10,
                text="Innovation Trigger",
                showarrow=False,
                arrowhead=0)
        fig.add_annotation(x=15, y=90,
                text="Peak of Inflation Expectations",
                showarrow=False,
                arrowhead=0)
        fig.add_annotation(x=20, y=10,
                text="Trough of Disillusionment",
                showarrow=False,
                arrowhead=0)
        fig.add_annotation(x=30, y=12,
                text="Slope of Enlightenment",
                showarrow=False,
                arrowhead=0)
        fig.add_annotation(x=45, y=17,
                text="Plateau of Productivity",
                showarrow=False,
                arrowhead=0)

        fig.update_xaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(title='Market Hype or Expectations [%]') 
        fig.update_xaxes(title='Time') 
        fig.update_traces(hovertext=hover_name, hovertemplate=f'<b>{hover_name}<b>') # provide hover that names curve

    elif dropdown_selection == "MA":
        'You have selected "{}"'.format(dropdown_selection)
        hover_name = "Market Adoption"
        fig = px.line(df_market_adoption, x='x1', y='y1', title = "Loonshot R&D Project:")
        fig.add_annotation(x=pos_x, y=market_adoption_curve_formula(pos_x),
                text="Loonshot R&D Project",
                showarrow=True,
                arrowhead=3)
        fig.add_vline(
            x=5, line_width=3, line_dash="dash", 
            line_color="grey")
        fig.add_vline(
            x=8, line_width=3, line_dash="dash", 
            line_color="grey")
        fig.add_vline(
            x=10, line_width=3, line_dash="dash", 
            line_color="grey")
        fig.add_vline(
            x=16, line_width=3, line_dash="dash", 
            line_color="grey")
        
        fig.add_annotation(x=2.5, y=80,
                text="Innovators",
                showarrow=False,
                arrowhead=0)
        fig.add_annotation(x=6.5, y=80,
                text="Early Adopters",
                showarrow=False,
                arrowhead=0)
        fig.add_annotation(x=9, y=80,
                text="Early Majority",
                showarrow=False,
                arrowhead=0)
        fig.add_annotation(x=13, y=80,
                text="Late Majority",
                showarrow=False,
                arrowhead=0)
        fig.add_annotation(x=20, y=80,
                text="Laggards",
                showarrow=False,
                arrowhead=0)
        fig.update_xaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(title='Market Adoption [%]') 
        fig.update_xaxes(title='Time') 
        fig.update_traces(hovertext=hover_name, hovertemplate=f'<b>{hover_name}<b>') # provide hover that names curve

        
    elif dropdown_selection == "MS":
        'You have selected "{}"'.format(dropdown_selection)
        hover_name = "Market Share"
        fig = px.line(df_market_share, x='x2', y='y2', title = "Loonshot R&D Project:")
        fig.add_annotation(x=pos_x, y=market_share_curve_formula(pos_x),
                text="Loonshot R&D Project",
                showarrow=True,
                arrowhead=3)
        
        fig.update_xaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(title='Market share [%]') # hide the y title
        fig.update_xaxes(title='') # hide the y title
        fig.update_traces(hovertext=hover_name, hovertemplate=f'<b>{hover_name}<b>') # provide hover that names curve

    elif dropdown_selection == "MC":
        'You have selected "{}"'.format(dropdown_selection)
        hover_name = "Market Action"
        fig = px.line(df_market_action, x='x3', y='y3', title = "Loonshot R&D Project:")
        fig.add_annotation(x=pos_x, y=market_action_curve_formula(pos_x),
                text="Loonshot R&D Project",
                showarrow=True,
                arrowhead=3)
        
        fig.update_xaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(title='Market Share [%]')
        fig.update_xaxes(title='Time')
        fig.update_traces(hovertext=hover_name, hovertemplate=f'<b>{hover_name}<b>') # provide hover that names curve

    elif dropdown_selection == "MK":
        'You have selected "{}"'.format(dropdown_selection)
        hover_name = "Market Knowledge"
        fig = px.line(df_market_knowledge, x='x4', y='y4', title = "Loonshot R&D Project:")
        fig.add_annotation(x=pos_x, y=market_knowledge_curve_formula(pos_x),
                text="Loonshot R&D Project",
                showarrow=True,
                arrowhead=3)
        
        fig.update_xaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(title='Market Knowledge [%]')
        fig.update_xaxes(title='Time')
        fig.update_traces(hovertext=hover_name, hovertemplate=f'<b>{hover_name}<b>') # provide hover that names curve

    elif dropdown_selection == "ER":
        'You have selected "{}"'.format(dropdown_selection)
        hover_name = "Emerging Risk in the Market"
        fig = px.line(df_emerging_risk, x='x5', y='y5', title = "Loonshot R&D Project:")
        fig.add_annotation(x=pos_x, y=emerging_risk_curve_formula(pos_x),
                text="Loonshot R&D Project",
                showarrow=True,
                arrowhead=3)
        
        fig.update_xaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(title='Emerging Risk in the Market [%]')
        fig.update_xaxes(title='Time')
        fig.update_traces(hovertext=hover_name, hovertemplate=f'<b>{hover_name}<b>') # provide hover that names curve

    elif dropdown_selection == "PM":
        'You have selected "{}"'.format(dropdown_selection)
        hover_name = "Innovation Perception Map"
        fig = px.scatter(df_perception_map, x='x6', y='y6', title = "Loonshot R&D Project:")
        fig.add_annotation(x=pos_x, y=innovation_perception_map_formula(pos_x),
                text="Loonshot R&D Project",
                showarrow=True,
                arrowhead=3)

        fig.add_vline(
            x=df_perception_map['x6'].max()/2, line_width=3, line_dash="dash", 
            line_color="grey")
        fig.add_hline(
            y=df_perception_map['y6'].max()/2, line_width=3, line_dash="dash", 
            line_color="grey")

        fig.add_annotation(x=df_perception_map['x6'].max()/4, y=df_perception_map['y6'].max()*3/4,
                text="Aggressively pursue",
                showarrow=False,
                arrowhead=0)

        fig.add_annotation(x=df_perception_map['x6'].max()*3/4, y=df_perception_map['y6'].max()*3/4,
                text="Build business support",
                showarrow=False,
                arrowhead=0)

        fig.add_annotation(x=df_perception_map['x6'].max()/4, y=df_perception_map['y6'].max()/4,
                text="Continue to investigate",
                showarrow=False,
                arrowhead=0)

        fig.add_annotation(x=df_perception_map['x6'].max()*3/4, y=df_perception_map['y6'].max()/4,
                text="Ignore",
                showarrow=False,
                arrowhead=0)

        fig.add_annotation(x=df_perception_map['x6'].min(), y=df_perception_map['y6'].min(),
                text="[Low, Low]",
                showarrow=False,
                arrowhead=0)

        fig.add_annotation(x=df_perception_map['x6'].max(), y=df_perception_map['y6'].min(),
                text="[High, Low]",
                showarrow=False,
                arrowhead=0)

        fig.add_annotation(x=df_perception_map['x6'].min(), y=df_perception_map['y6'].max(),
                text="[Low, High]",
                showarrow=False,
                arrowhead=0)

        fig.add_annotation(x=df_perception_map['x6'].max(), y=df_perception_map['y6'].max(),
                text="[High, High]",
                showarrow=False,
                arrowhead=0)
        
        fig.update_xaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(showticklabels=False) # hide the xticks
        fig.update_yaxes(title='Expected Benefit')
        fig.update_xaxes(title='Investment')
        fig.update_traces(hovertext=hover_name, hovertemplate=f'<b>{hover_name}<b>') # provide hover that names curve

    elif dropdown_selection == "FM":
        'You have selected "{}"'.format(dropdown_selection)
        hover_name = "Future Map Curve"

        if model_selection == "N": # Catch to "N" None Available selection in Select Model Menu dropdown
            model_selection = 'Regression'

        if g_imported == True:
            if default_file_name == cur_file_name:
                df_cur_dataframe = df_future_map.copy(deep=True)
                X = df_cur_dataframe.year.values.reshape(-1, 1)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, df_cur_dataframe.v, random_state=42)
                g_x_label = default_x_label
                g_y_label = default_y_label
                g_slider_max = max(df_cur_dataframe.year)
                g_slider_min = min(df_cur_dataframe.year)
                
            elif df_real_company_file_name == cur_file_name:
                df_cur_dataframe = df_imported.copy(deep=True)
                X = df_cur_dataframe.year_x.values.reshape(-1, 1)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, df_cur_dataframe.funding_amt, random_state=10)
                g_x_label = bio_firm_x_label
                g_y_label = bio_firm_y_label
                g_slider_max = max(df_cur_dataframe.year_x)
                g_slider_min = min(df_cur_dataframe.year_x)
                
            elif df_fake_company_file_name == cur_file_name:
                df_cur_dataframe = df_imported.copy(deep=True)
                X = df_cur_dataframe.Year_x.values.reshape(-1, 1)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, df_cur_dataframe.Funding_Amt, random_state=sample_amount)
                g_x_label = bio_firm_x_label
                g_y_label = bio_firm_y_label
                g_slider_max = max(df_cur_dataframe.Year_x)
                g_slider_min = min(df_cur_dataframe.Year_x)
                
        elif g_imported == False:
            df_cur_dataframe = df_future_map.copy(deep=True)
            X = df_cur_dataframe.year.values.reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(
                X, df_cur_dataframe.v, random_state=42)
            g_x_label = default_x_label
            g_y_label = default_y_label
            g_slider_max = max(df_cur_dataframe.year)
            g_slider_min = min(df_cur_dataframe.year)
        
        model = models[model_selection]()

        if not df_imported.empty:
            if default_file_name == cur_file_name:
                model.fit(X, df_cur_dataframe.v)
            elif df_real_company_file_name == cur_file_name:
                model.fit(X, df_cur_dataframe.funding_amt)
            elif df_fake_company_file_name == cur_file_name:
                model.fit(X, df_cur_dataframe.Funding_Amt)
        else:
            model.fit(X, df_cur_dataframe.v)

        x_range = np.linspace(X.min(), X.max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'))
        fig.add_trace(go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'))
        fig.add_trace(go.Scatter(x=x_range, y=y_range, name='prediction'))

        x_value = int(pos_x)
        
        global saved_y_pos
        
        if model_selection == 'Linear Regression':

            # Regression line y = mx + b
            m = model.coef_[0]
            b = model.intercept_ 
            
            y_value = (pos_x  * m) + b
            saved_y_pos = y_value

            # Display the project along the prediction line
            fig.add_annotation(x=x_value, y=y_value,
                text="Loonshot",
                showarrow=True,
                arrowhead=3)
        
        elif model_selection == 'Decision Tree Regression':

            y_value, saved_y_pos = track_along_curve_with_slider(x_value, saved_y_pos, x_range, y_range)

            # Display the project along the prediction curve
            fig.add_annotation(x=x_value, y=y_value,
                text="Loonshot",
                showarrow=True,
                arrowhead=3)

        elif model_selection == 'K-Nearest Neighbors Regression':

            y_value, saved_y_pos = track_along_curve_with_slider(x_value, saved_y_pos, x_range, y_range)
            
            # Display the project along the prediction curve
            fig.add_annotation(x=x_value, y=y_value,
                text="Loonshot",
                showarrow=True,
                arrowhead=3)

        elif model_selection == 'Logistic Regression':

            y_value, saved_y_pos = track_along_curve_with_slider(x_value, saved_y_pos, x_range, y_range)

            # Display the project along the prediction curve
            fig.add_annotation(x=x_value, y=y_value,
                text="Loonshot",
                showarrow=True,
                arrowhead=3)

        elif model_selection == 'Linear Discriminant Analysis':

            y_value, saved_y_pos = track_along_curve_with_slider(x_value, saved_y_pos, x_range, y_range)
            
            # Display the project along the prediction curve
            fig.add_annotation(x=x_value, y=y_value,
                text="Loonshot",
                showarrow=True,
                arrowhead=3)
        elif model_selection == 'Decision Tree Regression with AdaBoost':

            y_value, saved_y_pos = track_along_curve_with_slider(x_value, saved_y_pos, x_range, y_range)

            # Display the project along the prediction curve
            fig.add_annotation(x=x_value, y=y_value,
                text="Loonshot",
                showarrow=True,
                arrowhead=3)

        elif model_selection == 'Gaussian Naive Bayes':

            y_value, saved_y_pos = track_along_curve_with_slider(x_value, saved_y_pos, x_range, y_range)
            
            # Display the project along the prediction curve
            fig.add_annotation(x=x_value, y=y_value,
                text="Loonshot",
                showarrow=True,
                arrowhead=3)

        elif model_selection == 'Linear Support Vector Regression':

            y_value, saved_y_pos = track_along_curve_with_slider(x_value, saved_y_pos, x_range, y_range)

            # Display the project along the prediction curve
            fig.add_annotation(x=x_value, y=y_value,
                text="Loonshot",
                showarrow=True,
                arrowhead=3)

        # Display the x and y axis and associated labels
        # Display the hover text for the points and curves
        fig.update_xaxes(showticklabels=True) 
        fig.update_yaxes(showticklabels=True)
        fig.update_yaxes(title=g_y_label)
        fig.update_xaxes(title=g_x_label)
        fig.update_traces(hovertext=hover_name, hovertemplate=f'<b>{hover_name}<b>') # provide hover that names curve

    return fig


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
# As part of developing this code base
# Use for troubleshooting
def my_testing(my_tests_turn_on_or_off):
    if my_tests_turn_on_or_off == True:
        #Python versions
        print("Usage: Module Versions: \n")
        print('Python: {}'.format(sys.version))
        print('plotly: {}'.format(plotly.__version__))
        print('dash: {}'.format(dash.__version__))
        print('pandas: {}'.format(pandas.__version__))
        print('sklearn: {}'.format(sklearn.__version__))

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
# Entry Point
if __name__ == '__main__':

    # Python console display print on or off
    my_testing(False)

    # Run Dash
    app.run_server(debug=True, port=set_port)
