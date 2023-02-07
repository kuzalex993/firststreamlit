import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
#st.set_page_config(layout="wide")
st.title('California housing. Regression problem')

@st.cache
def load_data():
    df = pd.read_parquet('dataset/dataset.parquet')
    return df

def descript(df: pd.DataFrame) -> pd.DataFrame:
    naDf = df.isna().sum()
    naNull = df.isnull().sum()
    dfUnique = df.nunique()
    dsDf = pd.concat([naDf, naNull, dfUnique], axis = 1)
    dsDf.columns = ['# of N/A values', '# of NULL values', '# of unique values']
    return dsDf.transpose()

data = load_data()

##===========================================================

with st.expander("Open to see Raw Data"):
    st.subheader('Raw Data')
    st.write(data)

##===========================================================
st.subheader('Description statistic')
st.markdown('We see here some common statistic for each column of the dataset.')
st.write(data.describe())
st.caption('Description statistic')

##===========================================================
st.markdown("Let's check completeness of the data.")
st.markdown("Dataset does not contain NA and NULL values")
naDf = descript(data)
st.dataframe(naDf)

##===========================================================
st.subheader('Histogram for predictive variable.')
st.markdown("Here we are going to study predictive variable.")
st.markdown("First, we can build histogram")
fig_MHV, axes = plt.subplots(1, 1, figsize=(10, 8))
# Plot frequency plot/ histogram
sns.histplot(x="MedHouseVal", kde = True, data = data, ax = axes, bins = 40)
axes.set(xlabel="House Value", ylabel='Density')
axes.xaxis.label.set_size(18)
axes.yaxis.label.set_size(18)
axes.tick_params('y', labelsize = 14)
axes.tick_params('x', labelsize = 14)
st.pyplot(fig_MHV)
st.markdown('In the **MedHouseVal**, we see a large concentration of '
            'examples with a price of 500,000. This is probably '
            'a data collection problem. All the results which '
            "are above 500,000 were marked as 500,000. Let's look at all the rest variables.")
##===========================================================


col1, col2= st.columns(2)

with col1:
   k = 0
   for column in data.columns:
       if k % 2 != 0:
           fig, ax = plt.subplots()
           ax.hist(data[column], bins=50)
           plt.title('Histogram of {}'.format(column))
           plt.xlabel(column)
           st.pyplot(fig)
       k = k + 1
with col2:
    k = 0
    for column in data.columns:
        if k % 2 == 0:
            fig, ax = plt.subplots()
            ax.hist(data[column], bins=50)
            plt.title('Histogram of {}'.format(column))
            plt.xlabel(column)
            st.pyplot(fig)
        k = k + 1
st.markdown('The same situation with data **HouseAge**. '
            'All the houses which are older then 50 year '
            'are marked as 50 years old. In the future we can try to exclude '
            'this data from the train dataset.')

##===========================================================
st.subheader('Box-plots')
st.markdown('Now we can evaluate situation with outliers. '
            'From plots it is clear, there are big numbers of them. '
            'Later we can try to take data bellow some quantile: 90, 80, 75')
col1, col2= st.columns(2)
with col1:
   k = 0
   for column in data.columns:
       if k % 2 != 0:
           fig, ax = plt.subplots()
           ax.boxplot(data[column], patch_artist=True)
           plt.title('Boxplot of {}'.format(column))
           plt.xlabel(column)
           st.pyplot(fig)
       k = k + 1
with col2:
    k = 0
    for column in data.columns:
        if k % 2 == 0:
            fig, ax = plt.subplots()
            ax.boxplot(data[column], patch_artist=True)
            plt.title('Boxplot of {}'.format(column))

            plt.xlabel(column)
            st.pyplot(fig)
        k = k + 1


##===========================================================
st.subheader('Correlation matrix')
st.markdown('Now we can look at corelation matrix '
            'to understand whether we have some dependencies between variables or not.')
st.markdown(""" 
            There are two pairs with high correlation: 
            - Longitude and Latitude
            - AveRooms and AveBedrms.
            All dependencies can be logically explained. Here we have multicollinearity.
            """)
fig = plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap='plasma', annot=True, fmt='.2f')
st.pyplot(fig)

##===========================================================
st.subheader('Dependence prices to coordinates on the map')
st.markdown(""" 
            Now let's place the data about house prices on the map. Using information about 'Longitude', 'Latitude',
            'MedHouseVal' and 'Population' we build following map, where:
            - color of the circle is based on Mean House Price. A circle is more red as higher price.
            - size of the circle is based on Population. A circle is bigger as Population as bigger.
            """)
fig = px.scatter_mapbox(data, lat='Latitude', lon='Longitude', size='Population', size_max=15,
                         zoom=4.4, center=dict(lat=data['Latitude'].mean() + 1, lon=data['Longitude'].mean() - 1.5),
                        mapbox_style="stamen-terrain", template='plotly_dark', title='Dependence of the price on longitude, latitude',
                       color='MedHouseVal')
fig.update_layout(title_x=0.5, height=600)

st.plotly_chart(fig, theme=None, use_container_width=True)

st.subheader('Preliminary conclusions')
st.markdown(""" 
            At the moment we can make some notes:
            - several variables have collecting issues. *MedHouseVal* and *HouseAge* have high number of examples 
                at maximum border. Most probably we will need to handle with this data somehow.
            - we have quite high number outliers. If the quality of models will not be not satisfiable, 
                then later we can try to take data bellow some quantile: 90, 80, 75.
            - we have two pairs with high correlation: Longitude and Latitude, AveRooms and AveBedrms. 
                Here we have multicollinearity.
            - we see that the price depends on the district. Namely, from proximity to the coastal zone. 
                It decreases if we move closer to the ocean and decreases as we move inland or north.
            """)

