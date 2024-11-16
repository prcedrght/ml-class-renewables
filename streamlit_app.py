import streamlit as st
# from streamlit_option_menu import option_menu
import base64
# import plotly.express as px
import pandas as pd
# import nbformat

# with open('./data/eda_notebook.ipynb') as f:
#     notebook = nbformat.read(f, as_version=4)

# national_capacity = pd.read_csv("./data/national_generation_capacity_stacked_filtered.csv")
# national_capacity_cleaned = national_capacity[['technology', 'year', 'country', 'capacity']].copy()
# time_data = pd.read_csv("./data/time_series_60min_singleindex_filtered.csv")
# long_time_data = pd.read_pickle("./data/long_pickle.pkl")

final_head = pd.read_feather("./data/final_head.feather")
# Function to read and encode the image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to apply custom CSS
# def set_background(png_file, size='cover'):
#     bin_str = get_base64_of_bin_file(png_file)
#     css_str = f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/png;base64,{bin_str}");
#         background-size: {size};
#         background-repeat: no-repeat;
#         background-attachment: fixed;
#     }}
#     </style>
#     """
#     st.markdown(css_str, unsafe_allow_html=True)

# set_background("./images/renewables_collage.png", )

# # # Define the sidebar menu
# with st.sidebar:
#     selected = option_menu(
#         "Main Menu",
#         ["Introduction", "Data Gathering", "Data Prep/EDA", "PCA", "ARM", "Clustering", "Bayes", "Trees", "SVM", "Conclusion"],
#         icons=["house", "server", "table"],
#         menu_icon="cast",
#         default_index=0,
#     )
st.title("Smart Grid Load Prediction with Renewable Engergy Integration & Optimization")

intro, data, modeling, conclusion, references = st.tabs(["Introduction", "Data Prep & EDA", "Modeling", "Conclusion", "References"])

# Introduction Tab
with intro:
    st.header("Introduction")
    st.write("""#### _Can peak energy demands be optimized by understanding when renewable energy is most efficient?_""")
    st.image("./images/renewables_collage.png", use_column_width=True, caption='Renewable Energy as envisioned by GenAI')
    st.write("""Earth is in a crisis. Since the dawn of the industrial revolution modern society has become heavily dependent on the finite resources of fossil fuels which are a major contributor, if not the sole driver, behind modern [warming trends](https://bpb-eu-w2.wpmucdn.com/blogs.reading.ac.uk/dist/3/187/files/2020/01/lia_mwp-1.png). 
             This existential threat is true for every single person on Earth, no one will be able to totally avoid the turmoil this is causing. Heat waves, droughts, deep freezes, flooding, famine and diseases will all be inescapable. 
             While [experts suggest](https://fortune.com/2024/04/10/united-nations-climate-chief-humanity-has-2-years-to-save-the-world/) there is time to curb the effects of climate change, there is still significant effort needed to ensure a future for the next generations. 
             Thankfully a bright spot can be seen amongst the smog — the advancement of “green” energy sources and technologies. The ability to harness solar, wind, hydropower, or geothermal energy unlocks a paradigm that is rooted in clean, renewable power generation, and one that mitigates humanity’s impact on the planet. 
             These sources of energy are exciting but they come with shared challenges. 
             For example hydropower — like energy that comes from a dam or newer technology that harnesses the [kinetic energy of the tides and ocean currents](https://www.pnnl.gov/explainer-articles/tidal-energy#:~:text=What%20is%20tidal%20energy%3F,the%20water%20to%20move%20faster.) — requires significant infrastructure to move the energy captured from its point of origin to the consumer. 
             As of 2023, the Hoover Dam was capable of generating [1,080 megawatts](https://www.statista.com/statistics/1360603/power-capacity-of-the-hoover-dam/) at any given moment. 
             That is enough energy to power for roughly one million homes. However the dam is located in a relatively remote location (as are most dams), and Los Angeles which is the city that [consumes most of that energy (15%)](https://www.eia.gov/kids/for-teachers/field-trips/hoover-dam-hydroelectric-plant.php#:~:text=Nineteen%20percent%20of%20the%20electricity,that%20is%20270%20miles%20away.) is 270 miles away. 
             Aside from proximity challenges, each renewable energy source faces the challenge of relying on non-human factors like the weather — solar energy requires sunshine, wind requires wind, etc. 
             This means that renewable energy sources have a unique set of optimal weather conditions in which they can operate most efficiently.""")
    st.write("""With the invention of the internet and interconnectedness of things, also came the ability to better understand and monitor energy consumption on the electric grid. 
             This technology gave birth to the smart grid. A [smart grid relies on two-way communication](https://ieeexplore.ieee.org/document/8452969) between the power stations, batteries and the consumers to better utilize the energy available. 
             For example, if energy is in high demand, and customers are connected to a smart grid, energy companies can adjust thermostat settings to manage consumption and ensure blackouts do not occur. 
             There is complicated modeling involved in balancing the supply and demand, and renewable energy sources' notoriety for being sporadic make that more difficult. 
             By examining the relationship between renewable energy efficiency and grid demand, new evidence could be used in modeling to better plan for grid stress and ease for storage.""")
    st.image("./images/smart_grid.jpeg", use_column_width=True, caption='Smart Grids as the future of energy consumption imagined by AI.')
    st.write("""
    Machine learning (ML) techniques offer a means to insights in tackling these challenges by providing advanced tools for analyzing and optimizing the complex dynamics of renewable energy and smart grids. 
    While any given ML algorithms may not provide a clear answer, it can process vast amounts of data from various sources, such as weather forecasts, energy consumption patterns, and grid performance metrics, to predict energy production but more importantly allow for more questions and research to occur. 
    Predictive capability enable more efficient integration of renewable energy into the grid, minimizing waste and enhancing reliability. 
    Furthermore, ML can facilitate real-time decision-making and adaptive control strategies, allowing for dynamic adjustments to energy distribution and storage. 
    By leveraging these technologies, we can create a more resilient and sustainable energy infrastructure that not only meets current demands but also adapts to future challenges, ultimately contributing to a more sustainable and environmentally friendly energy landscape.""")
    st.write("""
In this study, the aim is to explore whether renewable energy generation has a relationship with grid demand, and if so how might energy providers enhance their smart grids with this knowledge. 
             Through focused machine learning algorithms designed to account for the inherent variability and unpredictability of renewable energy, models can be used to inform how authorities might optimize energy production and distribution. 
             Such models could significantly improve the efficiency of smart grids, reduce reliance on fossil fuels, and decrease the likelihood of energy shortages. 
             Ultimately, the research endeavors to contribute to a more sustainable energy infrastructure, one that not only meets the needs of today but anticipates and adapts to the challenges of tomorrow.
""")
    st.write("""
    #### 10 Questions to Answer:
    1. How well can renewable energy sources be integrated into meeting smart grid demands?
    2. What are the optimal weather conditions for renewable energy sources?
    3. Can we predict peak energy demand in smart grids based on both historical usage data and weather forecasts?
    4. Are there particular combinations of weather conditions that result in more efficient renewable energy generation?
    5. Are there particular combinations of weather conditions that result in more energy demand?
    6. Do particular climate regions have more efficient renewable energy generation?
    7. Do particular climate regions have more energy demand?
    8. How does seasonality impact regional energy demand and generation?
    9. How does time of day impact regional energy demand and generation?
    10. How much more renewable energy needs to be produced to meet the growing demand of energy with the AI boom?
    #
    All raw data and code can be found in this [GitHub repository](https://github.com/prcedrght/ml-class-renewables)
    """)

# Data Gathering Tab
with data:
    # st.header("Data Prep & EDA")
    sub_tabs = st.selectbox("Use dropdown for details:", ["Data Collection", "Data Cleaning", "Exploratory Data Analysis"])
    if sub_tabs == "Data Collection":
        st.write("""
        #### DATA COLLECTION
        In order to have a solid description of energy habits and production, a wide variety of data must be collected: consumption measurements for long term trends, time series consumption patterns, and weather history. 
        While there are several different renewable energy sources that should be considered for this investigation, 
        the sparsity of certain types like geothermal means this data is primarily made up of photovolatic, wind energy and hyrdo. 
                 Additionally, to help bound the research to a broad but manageable size, data was bucketed into the various EIA regions.
        The data used in this project was collected from the following sources: the [Meteostat](https://meteostat.net/en/), 
                 [National Renewable Energy Laboratory](https://www.nrel.gov/research/data-tools.html), 
                 and the [Energy Information Administration](https://www.eia.gov/). \n
        ##### Meteostat
        Meteostat is an open source platform that collects weather and climate data from all over the world. 
                The data is available via a [Python Library](https://pypi.org/project/meteostat/) and offers a simple mechanism to download extensive data. 
                By providing a series of geographic coordinations that mapped to the EIA regions, several meterological measurements are returned including but not limited to: average temperature, precipitation, wind speed and direction. \n
        ##### National Renewable Energy Laboratory
        The NREL is a government organization that provides a wealth of data on renewable energy sources. Because Meteostat does not have more recent data for solar irradiance, NREL was queried to pull in
                 a physical model of information about how much sunlight is hitting the Earth's surface. 
                 The [Physical Solar Model](https://developer.nrel.gov/docs/solar/nsrdb/psm3-5min-download/) was queried with same geographic coordinates and region mapping at over hour long intervals and summed for the day.\n
        ##### Energy Information Administration
        The EIA is a government organization that provides data on energy generation and consumption across all fuel types, providers and regions in the United States. 
                 The data was used to pull in information about energy consumption and generation by region. Generation data is provided at the daily level, however consumption is reported on the monthly level, needing to bring all datasets up to that granularity.
                 The [Electricity Data Browser](https://www.eia.gov/opendata/browser/electricity) has an excellent user friendly feature that allows you to generate the API call for your preferred method of downloading data.\n
        """)
    # st.write("Content for Data Gathering will go here.")

    if sub_tabs == "Data Cleaning":
        eia_head = pd.read_feather("./data/eia_head.feather")
        eia_head2 = pd.read_feather("./data/eia_head2.feather")
        
        st.write("""
        #### DATA CLEANING
        As mentioned earlier, all data needed to have some standard granularity and mapping keys to join them together. 
                 Each dataset was resampled up to the `Month` granularity and either averaging or summing values where appropriate. For example:""")
        st.code("""wind_monthly = region_climate.set_index('time').groupby(['location']).resample('MS')[['tavg','wspd','prcp']].mean().reset_index()""")
        st.write("""The Meteostat data was in relatively good shape with only a couple of dimensions missing some or all values, those were dropped from the dataset.
        The EIA dataset downloaded for Electricity Generation contained several fuel types that were not renewable -- such as coal and natural gas.
        """)
        st.dataframe(eia_head)
        st.write("""However, because the primary interest is in renewable energy, anything that wasn't Solar, Wind or Hydro was set aside to add in later for broader questions about meeting general demand and what percentage of renewables contribute.""")
        st.dataframe(eia_head2)
        st.write("""
        Similarly, the Consumption or Demand data from EIA contained records of energy consumption by sector types. In this case, the research was concerned with the total consumption (provided in miilions of kWh), so it was limited to `all sectors`.
        """)
        st.code("demand.sectorName.unique()")
        st.text("""array(['commercial', 'all sectors', 'transportation', 'residential',
       'other', 'industrial'], dtype=object)""")
        st.write("""In the Irradiance data collected from NREL, solar measurements are offered in three distinct values: Global Horizontal Irradiance (GHI), Direct Horizontal Irradiance (DHI), and Direct Normal Irradiance (DNI).
                 Each of these describe the amount of sunlight hitting the Earth's surface in different ways. GHI is the total amount of sunlight hitting the Earth's surface, DHI is the amount of sunlight that is diffused by the atmosphere, and DNI is the amount of sunlight that is hitting the Earth's surface directly.
                    These values were summed up to provide a total amount of sunlight hitting the Earth's surface in a given day a measure called `poa` for [Plane of Array](https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/plane-of-array-poa-irradiance/). 
                 Additionally, the date time was provided as separate columns, one for the year, month, day, and hour. These needed to be combined into a single datetime column for easier manipulation.""")
        st.code("""irradiance['poa'] = irradiance.GHI + irradiance.DHI + irradiance.DNI""")
        st.code("""rradiance['month_year']  = pd.to_datetime(irradiance[['Year', 'Month']].assign(day=1))""")
        st.write("""The various data sources were all joined together on the `EIA Region` and `Month` columns to create a single dataset that could be used for analysis. 
                 """)
        st.code("""final_df = pd.merge(green_pivot, region_demand, left_on=['date_time', 'respondent-name'], right_on=['date_time', 'eia_region'], how='inner').drop(columns=['respondent-name'])
final_df = pd.merge(final_df, wind_monthly, left_on=['date_time', 'eia_region'], right_on=['time', 'location'], how='inner').drop(columns=['location', 'time'])
final_df = pd.merge(final_df, irradiance_resampled, on=['date_time', 'eia_region'], how='inner').drop(columns=['month_year'])
final_df.head()""")
        st.dataframe(final_head)

    if sub_tabs == "Exploratory Data Analysis":
        st.write("""
        ## Renewable Energy Generation
        To get a sense of how much Megawatt Hours of energy are generated by each renewable source, first it was important to understand the general distribution of these sources.""")
        st.image("./images/MWh_Green_type.png", use_column_width=True, caption='In the US, Wind tends to produce more energy than Solar and Hyrdo.')
        st.write("""However, these sources produce a smaller amount of energy comparable all fuel types.""")
        st.image("./images/MWh_Fuel_type.png", use_column_width=True,)
        st.write("""Of course, energy is not produced all at once, there are fluctations throughtout the year, so it is important to view the production trends over time. This will likely be an important feature in producing a model to optimize the smart grid.""")
        st.image("./images/MWh_Green_type_time.png", use_column_width=True, caption='There is some clear seasonality to the amount of energy generated throughout the year.')
        st.write("""## Energy Demand""")
        st.write("""The same can be said for energy demand, there are fluctuations throughout the year, so it will be important to understand how these two features interact.
                 """)
        st.image("./images/MWh_demand2.png", use_column_width=True, caption='In addition to regional nuances like population size and energy habits, some regions have more demand than others because of the size of those regions.')
        st.write("""## Climate & Irradiance""")
        st.write("""As one might expect, climate data is highly seasonal as well. When examining temperature and irradiance data, there are clear peaks in the summer months and valleys throughout the winter months in the Northenr Hemisphere.""")
        st.image("./images/avg_temp.png", use_column_width=True, caption='There are clear patterns that align with expectations. For example, the North West region has some of the coldest temperatures throughtout the year.')
        st.image("./images/irradiance.png", use_column_width=True, caption='Irradiance is a measure of the amount of sunlight hitting the Earth\'s surface.')
        st.write("""Unlike temperature and irradiance trends, wind speed and precipitation have less decernable seasonal patterns. This is likely due to the fact that these measurements are more dependent on local weather patterns and less on the time of year.""")
        st.image("./images/avg_wind.png", use_column_width=True, caption='The Central region tends to have the highest average wind speeds, most likely because of the vast flat terrain which would not impede wind flow.')
        st.image("./images/avg_rain.png", use_column_width=True, caption='In the winter months of 2022 and 2023, California experienced some record precipitation which can be observed here in this graph. Of course, what is not accounted for would be all of the snow that fell that winter as well.')

    # # Sample data
    # df = pd.DataFrame({
    #     "Fruit": ["Apples", "Oranges", "Bananas", "Grapes"],
    #     "Amount": [10, 15, 7, 12]
    # })

    # # Create a Plotly figure
    # fig = px.bar(df, x="Fruit", y="Amount", title="Fruit Amounts")

    # # Display the Plotly figure in Streamlit
    # st.plotly_chart(fig)

    # # Add some text to the Streamlit app
    # st.write("This is a bar chart showing the amount of different fruits.")

with modeling:
    
    sub_tabs = st.selectbox("Use dropdown to explore different models:", ["PCA", "Clustering", "ARM", "Naïve Bayes", 'Decision Trees', 'Regression', 'SVM', 'Ensemble Learning'])
    if sub_tabs == "PCA":
        st.write("""All code can be found in this [notebook](https://github.com/prcedrght/ml-class-renewables/blob/main/data/eda_notebook.ipynb).""")
        pca_head = pd.read_feather("./data/pca_head.feather")
        st.title("PCA")
        st.write("""
        #### Principal Component Analysis
        Principal Component Analysis (PCA) is a technique used to emphasize variation and bring out strong patterns in a dataset. 
                 It simplifies the complexity of high-dimensionality data while preserving as much variabilty as possible. 
                 At it's core, PCA changes the original varaiables through orthogonal linear transfromation into prinicpal components by creating a new set of uncorrelated varibles that are ordered by the amount variance captured.
                 This allows the data to be visualized at a lower dimensionality and can be used to identify patterns in the data that may not be immediately obvious.
                 """)
        st.dataframe(final_head)
        st.write("""Becuase the data contains dimension that are not quantitative, those must be removed before trasnforming it. Additionally, as a final preprocessing step, the date column was encoded into two `sin` and `cos` cyclical features.
                 This helps ensure the models know that December is closer to January than it is to June, otherwise it would assume that `12` is bigger than `1` not next to each other.""")
        st.code("""from sklearn.preprocessing import FunctionTransformer

def sin_transformer(period: int) -> FunctionTransformer:
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period: int) -> FunctionTransformer:
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

def create_sine_cosine_doy_feature(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df['month'] = df[date_col].dt.month
    df['month_sin'] = sin_transformer(12).fit_transform(df['month'])
    df['month_cos'] = cos_transformer(12).fit_transform(df['month'])
    return df""")
        st.image('./images/sin_cos.png', use_column_width=True, caption='Sin and Cosine Encoding')
        st.write("""The final dataset used for PCA is shown below with the qualitative and target column removed, along with the date column encoded to represent the cyclical nature of time.""")
        st.dataframe(pca_head)
        st.write("""The `StandardScaler()` returns an array with variables encoded between -1 and 1.""")
        st.text("""array([[ 1.06487881e-01,  3.09447229e-01, -8.68151396e-02, ...,
        -3.57169225e-01,  7.15290024e-01,  1.23094752e+00],
       [-6.68569182e-04, -4.52746749e-01, -6.75608019e-01, ...,
        -6.80527543e-01,  7.15290024e-01,  1.23094752e+00],
       [-2.46356106e-01, -4.65672359e-01,  2.44937398e+00, ...,
        -8.63650010e-01,  7.15290024e-01,  1.23094752e+00],
       ...,
       [-4.86251401e-01, -3.04727678e-01, -2.98561589e-01, ...,
        -7.43367457e-01,  7.63596373e-03,  1.42028015e+00],
       [ 2.89908831e-01, -4.66506269e-01, -6.74371801e-01, ...,
        -1.44779499e+00,  7.63596373e-03,  1.42028015e+00],
       [-5.59861125e-01, -2.73456043e-01,  4.91381587e-01, ...,
        -8.79570660e-01,  7.63596373e-03,  1.42028015e+00]])""")
        st.write("""After scaling the data, and fit transforming the data it can be plotted for both `n=2` and `n=3` components.""")
        st.image("./images/pca_2.png",)
        st.image("./images/pca_3.png",)
        st.write("""Using a PCA of `n=2` components, only `~48%` of the variance is data can be explained.""")
        st.text('array([0.29927026, 0.19274337])')
        st.write("""While using PCA of `n=3` components, `~65%` of the variance is data can be explained.""")
        st.text('array([0.29927026, 0.19274337, 0.15979564])')
        st.write("""While this is a good start, in order to get to 95% of the explained variance, `n=6` components are required.""")
        st.image('./images/cumulative_variance.png', use_column_width=True, caption='The cumulative variance plot shows that 95% of the variance is explained by 7 components.')
        st.write("""The top three eigenvalues of this data are:""")
        st.text('[2.69345609 1.73470563 1.4381734]')
    
    if sub_tabs == "Clustering":
        st.write("""All code can be found in this [notebook](https://github.com/prcedrght/ml-class-renewables/blob/main/data/eda_notebook.ipynb).""")
        st.title("Clustering")
        st.write("""There are three clustering approaches that were used to determine patterns in the data: KMeans, Hierarchical Clustering, and DBSCAN.
                 KMeans is a partitioned clustering method that groups data into `k` distinct, non-overlapping subsets designed to minimize the variance within each cluster and maximize the variance between them.
                 Hierarchical Clustering builds a tree or dendrogram of clusters which can be cut at any point to determine the number of clusters. These can be created bottom-up (agglomerative) or top-down (divisive).
                 Lastly, DBSCAN (Density-Based Spatial Clustering of Application with Noise) is an approach that groups points together that are close in proximity while marking low-density regions as outliers.""")
        st.write("""
                #### Data Prep
                
                For KMeans, data preparation for this clustering method is the same as prepping for PCA using `n=3` components. Quantitative and target columns were removed, and the date column was encoded into cyclical features.""")
        st.write(""" 
        ### KMeans
        
        #### Silhouette Scores
        First to determine how many clusters are appropriate, the Silhouette Scores were calculated for a range of cluster sizes.""")
        st.code("""k_values = range(2, 11)
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pc_3)
    score = silhouette_score(pc_3, kmeans.labels_)
    silhouette_scores.append(score)""")
        st.image("./images/silhouette_scores.png", use_column_width=True, caption='The Silhouette Score plot highlighting 3 clusters is the best choice but should also look at 2 and 7 clusters.')  
        st.write("""Based on the plot above there are three possible `k` values to try: 2, 3, and 7. These three values are the closest to 1 eventhough they are quite far from it.""")
        st.write("""The centroids of the `k=2` clusters are as follows:""")
        st.text("""[[ 1.33158106 -0.01165726 -0.01272154]
 [-1.49717568  0.01310695  0.01430359]]""")
        st.image("./images/kmeans2_3d.png", use_column_width=True, caption='Centroids for k=2 are extremely close together and near the origin.')
        st.write("""The centroids of the `k=3` clusters are as follows:""")
        st.text("""[[ 1.45665848  0.05309448 -0.12386596]
 [-1.18628448 -2.85873459  0.8976224 ]
 [-1.33431624  0.57160266 -0.06205635]]""")
        st.image("./images/kmeans3_3d.png", use_column_width=True, caption='The 3D plot of the clusters shows that they are not easily interpretable except that there is clear concentration across the X & Y axes with dispersion across the Z.')
        st.write("""The centroids of the `k=7` clusters are as follows:""")
        st.text("""[[ 0.53972789  0.86615518  0.14313369]
 [ 1.2560246  -0.21309264 -1.1325972 ]
 [-1.2502602  -0.09196156 -0.93727056]
 [ 2.16392606 -0.37512082  0.8736565 ]
 [-1.23452401 -2.95152657  1.0130149 ]
 [-2.16308385  0.89938247  0.66314643]
 [ 0.09176567  2.0939616   2.20355834]]""")
        st.image("./images/kmeans7_3d.png", use_column_width=True,)
        st.write("""Because it is hard to distinguish the centroids in the 3D plot, the seven clusters were also plotted and colored by their respective clusters to help distinguish the groupings in the data.""")
        st.image("./images/kmeans7_3d2.png", use_column_width=True)
        st.write("""
        ### Hierarchical Clustering
        This extensive dataset is too large to produce a dendrogram without more compute resources, so the data was sampled and then clustered. For this clustering, `scipy`, the `linkage` function, and the `ward` method were used to generate these insights. 
                 According to this method, the most logical number of clusters that should be used is `n=2` because there is one leg that has significant distance compared to the others.
                 But the dendrogram also suggests that 3, 5 or even 7 could be used as well. KMeans clustering is highly dependent on the placement of the centroids, and like the plots show, there is some overlap between these clusters.
                 However, climate data could be considered hierarhcial in nature. For example, temperature could be independent on wind speed but temperature is also influenced by the amount of sunlight hitting the Earth.
        """)
        st.image("./images/dendrogram.png", use_column_width=True, caption='The Dendrogram plot shows that there is a long tail of groupings.')
        st.write("""
        ### DBSCAN
        Examining the DBSCAN plot, there is a large homogenous cluster with several other smaller clusters that are quite sparse.
        Similar to the hierarchical clustering method, the DBSCAN method is suggesting that a higher order of clusters -- in this case `n=6` -- would be the most appropriate for this dataset.
        """)
        st.image("./images/dbscan.png", use_column_width=True)
        st.write("""
        ### Clustering Conclusions
        
        Because climate data is of high dimensionality, contains lots of noise, and often have complex interactions, it is not surprising that clustering methods did not produce clear patterns amongst the data.
                 With that said, these methods have provided some insight into how one might group the data with regards to the Energy Demand targets. While it is simple enough to say that energy consumption may be high or low based on the weather, it is in fact a complex interaction and likely should be modeled as a continuous variable.
        """)
        

    if sub_tabs == "ARM":
        st.write("""All code can be found in this [notebook](https://github.com/prcedrght/ml-class-renewables/blob/main/data/eda_notebook.ipynb).""")
        tmp_arm_head = pd.read_feather("./data/tmp_arm_head.feather")
        tmp_arm_head2 = pd.read_feather("./data/tmp_arm_head2.feather")
        final_arm = pd.read_feather("./data/final_arm_df.feather")
        st.title("Association Rule Mining")
        st.image("./images/arm_concept.jpeg", use_column_width=True)
        st.write("""
        Association Rule Mining is a technique used in data mining to discover interesting relationships, patterns, or associations among a set of items in large databases. It is commonly used in market basket analysis to identify sets of products that frequently co-occur in transactions.
                 In ARM, there are rules. These associations are an implicit suggestion that if A occurs then B is also likely to occur.
                 The Apriori Algorithm is used to find frequent itemsets and generate those association rules.
                 In general, it generates a candidate itemsets while using a support threshold to filter out infrequent items.
                 Then it prunes the itemsets that do meet the minimum support threshold.
                 Next, it combines itemsets to generate new candidate itemsets.
                 Then this process repeats until no more frequent itemsets can be found and generates the rules based on a minimum confidence threshold.
                 The support of an itemset is the proportion of transactions in the database in which the itemset appears, while confidence is a measure of the reliability of an association rule. It is the proportion of transactions containing itemset A that also contain itemset B.
                 """)
        st.image("./images/arm_formulas.png", use_column_width=True, caption='Apriori Concepts borrowed from https://medium.com/@nrmnbabalik/apriori-algorithm-in-recommendation-systems-782e7cd83440')
        st.write("""### Data Prep""")
        st.write("""Because the Apriori method requires transactional data, and the climate-energy data set is in record format, it has to be manipulated into a different form. Additionally, because the majority of variables are quantiative in nature, those needed to be transformed into some discrete format.
                 As a reminder, the dataset used in this research looks like this:""")
        st.dataframe(final_head)
        st.write("""From here, the quantiative needed to be discretized using the following code:""")
        st.code("""arm_df['month'] = arm_df.month_year.dt.month
arm_df['year'] = arm_df.month_year.dt.year
arm_df['hydro_bin'] = pd.cut(arm_df.Hydro, bins=7, labels=['Hydro_Low', 'Hydro_Low-Medium', 'Hydro_Medium', 'Hydro_Medium-High', 'Hydro_High', 'Hydro_Very High', 'Hydro_Extremely High'])
arm_df['solar_bin'] = pd.cut(arm_df.Solar, bins=7, labels=['Solar_Low', 'Solar_Low-Medium', 'Solar_Medium', 'Solar_Medium-High', 'Solar_High', 'Solar_Very High', 'Solar_Extremely High'])
arm_df['wind_bin'] = pd.cut(arm_df.Wind, bins=7, labels=['Wind_Low', 'Wind_Low-Medium', 'Wind_Medium', 'Wind_Medium-High', 'Wind_High', 'Wind_Very High', 'Wind_Extremely High'])
arm_df['tavg_bin'] = pd.cut(arm_df.tavg, bins=7, labels=['Tavg_Low', 'Tavg_Low-Medium', 'Tavg_Medium', 'Tavg_Medium-High', 'Tavg_High', 'Tavg_Very High', 'Tavg_Extremely High'])
arm_df['wspd_bin'] = pd.cut(arm_df.wspd, bins=7, labels=['Wspd_Low', 'Wspd_Low-Medium', 'Wspd_Medium', 'Wspd_Medium-High', 'Wspd_High', 'Wspd_Very High', 'Wspd_Extremely High'])
arm_df['prcp_bin'] = pd.cut(arm_df.prcp, bins=7, labels=['Prcp_Low', 'Prcp_Low-Medium', 'Prcp_Medium', 'Prcp_Medium-High', 'Prcp_High', 'Prcp_Very High', 'Prcp_Extremely High'])
arm_df['poa_bin'] = pd.cut(arm_df.poa, bins=7, labels=['POA_Low', 'POA_Low-Medium', 'POA_Medium', 'POA_Medium-High', 'POA_High', 'POA_Very High', 'POA_Extremely High'])
arm_df = arm_df.drop(columns=['month_year', 'Hydro', 'Solar', 'Wind', 'tavg', 'wspd', 'prcp', 'poa'])""")
        st.write("""This discretized data looks like the follwoing:""")
        st.dataframe(tmp_arm_head)
        st.write("""Because Python does not handle transactional data well for Apriori algorithms, the data has to be transformed one last time where each item is encoded as its own dimension.""")
        st.dataframe(final_arm)
        st.write("""### Results""")
        st.write("""
#### Top 15 Rules for Support
""")
        st.text("""support                          itemsets
0   0.831979                       (Solar_Low)
1   0.688196                       (Hydro_Low)
2   0.685299                        (Wind_Low)
3   0.573511             (Solar_Low, Wind_Low)
4   0.569347            (Solar_Low, Hydro_Low)
5   0.477677             (Hydro_Low, Wind_Low)
6   0.391756  (Solar_Low, Hydro_Low, Wind_Low)
7   0.352286                        (Prcp_Low)
8   0.326287                     (Wspd_Medium)
9   0.307481                     (Prcp_Medium)
10  0.303255                 (Wspd_Low-Medium)
11  0.279351          (Solar_Low, Prcp_Medium)
12  0.266638             (Solar_Low, Prcp_Low)
13  0.263389          (Solar_Low, Wspd_Medium)
14  0.261505           (Wspd_Medium, Wind_Low)""")
        st.write("""
#### Top 15 Rules for Confidence
""")
        st.text("""antecedents             consequents  confidence
0            (Prcp_Medium)             (Solar_Low)    0.908516
1               (Wind_Low)             (Solar_Low)    0.836877
2              (Hydro_Low)             (Solar_Low)    0.827303
3    (Hydro_Low, Wind_Low)             (Solar_Low)    0.820127
4            (Wspd_Medium)             (Solar_Low)    0.807232
5            (Wspd_Medium)              (Wind_Low)    0.801457
6               (Prcp_Low)             (Solar_Low)    0.756879
7               (Wind_Low)             (Hydro_Low)    0.697034
8              (Hydro_Low)              (Wind_Low)    0.694100
9              (Solar_Low)              (Wind_Low)    0.689334
10  (Solar_Low, Hydro_Low)              (Wind_Low)    0.688079
11             (Solar_Low)             (Hydro_Low)    0.684329
12   (Solar_Low, Wind_Low)             (Hydro_Low)    0.683083
13              (Wind_Low)  (Solar_Low, Hydro_Low)    0.571656
14             (Hydro_Low)   (Solar_Low, Wind_Low)    0.569250""")
        st.write("""
#### Top 15 Rules for Lift
""")
        st.text("""
               antecedents             consequents  confidence   support  \\
0            (Wspd_Medium)              (Wind_Low)    0.801457  0.261505   
1            (Prcp_Medium)             (Solar_Low)    0.908516  0.279351   
2              (Hydro_Low)              (Wind_Low)    0.694100  0.477677   
3               (Wind_Low)             (Hydro_Low)    0.697034  0.477677   
4              (Solar_Low)              (Wind_Low)    0.689334  0.573511   
5               (Wind_Low)             (Solar_Low)    0.836877  0.573511   
6   (Solar_Low, Hydro_Low)              (Wind_Low)    0.688079  0.391756   
7               (Wind_Low)  (Solar_Low, Hydro_Low)    0.571656  0.391756   
8              (Hydro_Low)             (Solar_Low)    0.827303  0.569347   
9              (Solar_Low)             (Hydro_Low)    0.684329  0.569347   
10   (Solar_Low, Wind_Low)             (Hydro_Low)    0.683083  0.391756   
11             (Hydro_Low)   (Solar_Low, Wind_Low)    0.569250  0.391756   
12   (Hydro_Low, Wind_Low)             (Solar_Low)    0.820127  0.391756   
13           (Wspd_Medium)             (Solar_Low)    0.807232  0.263389   
14              (Prcp_Low)             (Solar_Low)    0.756879  0.266638   

        lift  
0   1.169499  
1   1.091994  
2   1.012841  
3   1.012841  
4   1.005887  
5   1.005887  
6   1.004056  
7   1.004056  
8   0.994380  
9   0.994380  
10  0.992570  
11  0.992570  
12  0.985755  
13  0.970255  
14  0.909733  
""")
        st.write("""
A minimum support threshold of `0.25` and confidence threshold of `0.5` were chosen because of the complexity, and likely sparseness of relationships between the variables. 
The results above seems to suggest this as well where rules tend to occur infrequently but when the antecedent occurs it is frequently found with its consequent. 
For example, looking at the top rule by `Lift`, there is a strong association between Medium Wind Speed `Wspd_Medium` and Low Wind Generation `Wind_Low`. 
While this rule occurs only a little more than one quarter of the time in the data, eight out of those 10 times they'll occur together.
However, there are only 7 rules that have a lift greater than 1 which means more often than not there is no association for the others.
### Conclusion
All in all, the ARM method proved useful in identifying some possible explanation to what other models are picking up on in terms of relationships between the variables.
It would seem energy generation between different months of the year are more closely related than the weather, perhaps this is because of the seasonality to generation and weather.
""")
        st.image("./images/association_rules.png", use_column_width=True, caption='The Association Rules plot shows that there are only 7 rules amongst 6 unique items that have a strong likelihood of occuring together.')
    if sub_tabs == "Naïve Bayes":
        st.write("""All code can be found in this [notebook](https://github.com/prcedrght/ml-class-renewables/blob/main/data/supervised_learning.ipynb).""")
        multinomial_nb_train = pd.read_feather("./data/mnb_train_df.feather")
        multinomial_nb_test = pd.read_feather("./data/mnb_test_df.feather")
        gaussian_train = pd.read_feather("./data/gnb_x_train.feather")
        gaussian_test = pd.read_feather("./data/gnb_x_test.feather")
        bernoulli_train = pd.read_feather("./data/bnb_x_train.feather")
        bernoulli_test = pd.read_feather("./data/bnb_x_test.feather")
        final_head = pd.read_feather("./data/final_head.feather")
        st.title("Naïve Bayes")
        st.write("""
        Naïve Bayes is a classification technique based on Bayes' Theorem with an assumption of independence between predictors. 
        In simple terms, a Naïve Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. 
        Even if these features depend on each other or upon the existence of the other features, all of these properties contribute to the probability of the class label.
        Naïve Bayes is especically useful for several reasons, it is simple, fast, can be used for binary or multiclass classification, and offers interpretability.
        Because the underlying distribution of the features determines whether a Gaussian, Multinomial, Bernoulli or Categorical model is used, the data needs to be of the same feature distribution.
        For example, if the data had discrete counts in the features then a Multinomial model would be most appropriate but if those features were binary then a Bernoulli model would be used.
        With that said, there are ways around this by using a custom model, feature transformation or separate models and combining them.
        This section will focus on three types of Naïve Bayes models: Multinomial, Gaussian, and Bernoulli.
        """)
        st.write("""
        ## Data Prep
        #### Multinomial Naïve Bayes
        In order to prepare the data for this particular model, the continuous variables such as `tavg`, `wspd`, `prcp`, and `poa` were discretized into 5 bins each.
        The categorical variables (`month` and `eia_region`) were encoded into dummy variables such that each unique value of those cateogries have their own column.
        """)
        st.dataframe(final_head)
        st.write("""The data above was transformed into below and split into training and testing sets. It is vital that these training and testing sets are separate from each other because using training data in your testing set would result in artifically higher accuracy scores.""")
        st.write("Mulitnomial Training Set")
        st.dataframe(multinomial_nb_train)
        st.write("Mulitnomial Testing Set")
        st.dataframe(multinomial_nb_test)
        st.write("""
        #### Gaussian Naïve Bayes
        Similarly, the original data was prepared for a Gaussian classifier by encoding the categorical variables and splitting the data into training and testing sets. 
        Continuous variables did not need to be transformed other than being normalized because Gaussian models presume those variables are normally distributed. This prep created sparse matricies to run the model on.""")
        st.write("Gaussian Training Set")
        st.dataframe(gaussian_train)
        st.write("Gaussian Testing Set")
        st.dataframe(gaussian_test)
        st.write("""
        #### Bernoulli Naïve Bayes
        For the final NB model, the data was prepared by encoding the categorical variables, binarizing the continuous variables, and splitting the data into training and testing sets. This prep created sparse matricies to run the model on.""")
        st.write("Bernoulli Training Set")
        st.dataframe(bernoulli_train)
        st.write("Bernoulli Testing Set")
        st.dataframe(bernoulli_test)
        st.write("""
        ## Results
        #### Multinomial Naïve Bayes
        This model returned an accuracy score of `~0.63` on the testing set. In particular, the model was relatively successful at predicting the `High` and `Low` demand labels.
        """)
        st.text("""
        Classification Report:
              precision    recall  f1-score   support

        High       0.75      0.71      0.73      7421
         Low       0.68      0.74      0.71      6678
 Medium High       0.56      0.54      0.55      7400
  Medium Low       0.51      0.52      0.52      6897

    accuracy                           0.63     28396
   macro avg       0.63      0.63      0.63     28396
weighted avg       0.63      0.63      0.63     28396
""")
        st.image("./images/mnb_confusion_matrix.png", use_column_width=True, caption='The Confusion Matrix shows that the model is most successful at predicting High energy demand.')
        st.write("""
        #### Gaussian Naïve Bayes
        This model was a poor performer with only an accuracy of `~0.48` on the testing set. When examing the confusion matrix, it becomes clear why. The training set in this model lack sufficient data for `Medium High` and `Medium Low` demand labels.""")
        st.text("""
        Classification Report:
              precision    recall  f1-score   support

        High       0.51      0.98      0.67      7421
         Low       0.47      0.97      0.63      6678
 Medium High       0.00      0.00      0.00      7400
  Medium Low       0.00      0.00      0.00      6897

    accuracy                           0.48     28396
   macro avg       0.24      0.49      0.32     28396
weighted avg       0.24      0.48      0.32     28396
        """)
        st.image("./images/gnb_confusion_matrix.png", use_column_width=True, caption='The Confusion Matrix shows that the model is not predicting Medium High and Medium Low energy demand.')
        st.write("""
        #### Bernoulli Naïve Bayes
        This model was the highest performer of the chose Naïve Bayes models with an accuracy of `~0.70` on the testing set. 
        Not only did it outperform the Multinomial model by a couple points, but it also managed to bring up the `f1-score` for the `Medium High` and `Medium Low` demand labels.""")
        st.text("""
        Classification Report:
              precision    recall  f1-score   support

        High       0.79      0.78      0.79      7421
         Low       0.76      0.76      0.76      6678
 Medium High       0.65      0.63      0.64      7400
  Medium Low       0.60      0.63      0.61      6897

    accuracy                           0.70     28396
   macro avg       0.70      0.70      0.70     28396
weighted avg       0.70      0.70      0.70     28396""")
        st.image("./images/bnb_confusion_matrix.png", use_column_width=True, caption='The Confusion Matrix shows that the Bernoullli model was more accurate for all of the given demand labels.')
        st.write("""
        ## Conclusion
        These various Naïve Bayes models performed moderately well given the features used. This means that these features have some predictive power in determining the energy demand.
        This passes a logical sanity check considering features like temperature should have a direct relationship with energy demand. For example, when it is hotter outside, people will be using more energy to cool their homes and offices.
        To improve on these models, more work can be done to reduce the colinearity between certain features, as well as ensuring a more appropriate distribution of labels end up in the training sets.
        """)
    if sub_tabs == 'Decision Trees':
        dt_test = pd.read_feather("./data/dt_x_test.feather")
        dt_train = pd.read_feather("./data/dt_x_train.feather")
        st.write("""All code can be found in this [notebook](https://github.com/prcedrght/ml-class-renewables/blob/main/data/supervised_learning.ipynb).""")
        st.title("Decision Trees")
        st.image("./images/tree_diagram_ex.png", use_column_width=True, caption='A simple decision tree diagram borrowed from https://insidelearningmachines.com/interpret_decision_trees/')    
        st.write("""
        Decision Trees are a non-parametric supervised learning method used for classification and regression. 
        The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
        They work by splitting the data into subsets based on the value of input features, creating a tree-like model of decisions. 
        Each internal node of the tree represents a decision based on an attribute, each branch represents the outcome of the decision, and each leaf node represents a class label (in classification) or a continuous value (in regression).
        The deeper the tree, the more complex the decision rules. This means that the model is more likely to overfit the data because 
                 it is generally possible to create an infinite number of trees. There are many possible ways
                  to split the data at each node, 
                 especially if continuous attributes are involved, and ach split can be based on different thresholds or combinations of features. 
                 Furthermore, even slight variations in the order of splits or the depth of the tree can lead to different tree structures. 
                 This flexibility, while powerful, is what can lead to overfitting if not properly controlled through techniques like pruning or setting a maximum tree depth.""")
        st.image("./images/CART_tree_titanic_survivors.png", use_column_width=True, caption='A decision tree diagram made with the Titanic Survivors dataset.')
        st.write("""### GINI, Entropy, and Information Gain
        \n
Gini Impurity, Entropy and Information gain are metrics used to evaluate how "goodness" of a split in a Decision Tree. 
"Goodness" in this sense means how well the model is splitting in each node to maximize the separation of class labels.
Gini Impurity measures the probablity that a randomly chosen element from the set would be incorrectly labeled 
if it was randomly labeled according to the distribution of labels in the subset.""")
        st.latex(r"""
        Gini = 1 - \sum_{i=1}^{n} p_i^2""")
        st.latex(r"""
        Entropy = -\sum_{i=1}^{n} p_i \log_2(p_i)""")
        st.latex(r"""
        \text{Information Gain}(D, A) = \text{Entropy}(D) - \sum_{v \in \text{Values}(A)} \frac{|D_v|}{|D|} \text{Entropy}(D_v)
                 """)
        st.write("""
        #### Consider this simple example regarding whether a child will play outside or not based on the weather conditions:
                 \n
        | Weather | Play |
        |---------|------|
        | Sunny   | Yes  |
        | Rainy   | No   |
        | Overcast| Yes  |
        | Sunny   | No   |
        | Overcast| No   |
                 \n
        Calculating the Entropy for the entire dataset:             
""")
        st.latex(r"""
        ( p(\text{Yes}) = \frac{2}{5} ) \\~\\
        ( p(\text{No}) = \frac{3}{5} ) \\~\\
        Entropy = -\left( \frac{2}{5} \log_2 \frac{2}{5} + \frac{3}{5} \log_2 \frac{3}{5} \right) =0.971
""")
        st.write("""Calculate Entropy for each attribute:""")
        st.latex(r"""
        \text{Sunny} = -\left( \frac{1}{2} \log_2 \frac{1}{2} + \frac{1}{2} \log_2 \frac{1}{2} \right) = 1 \\~\\
        \text{Rainy} = -\left(0 \log_2 0 + 1 \log_2 1\right) = 0 \\~\\
        \text{Overcast}) = -\left( \frac{1}{2} \log_2 \frac{1}{2} + \frac{1}{2} \log_2 \frac{1}{2} \right) = 1
""")    
        st.write("""Calculate Information Gain (IG) for each attribute:""")
        st.latex(r"""
        IG(Entropy, \text{Weather}) = 0.971 - \left( \frac{2}{5} \times 1 + \frac{1}{5} \times 0 + \frac{2}{5} \times 1 \right) \\~\\
        IG(Entropy, \text{Weather}) = 0.971 - 0.8 = 0.171
""")
        st.write("""
""")
        st.write("""-----""")
        st.write("""
        ## Data Prep
        To prepare the data for a decision tree, the categorical variables were encoded into dummy variables, 
                 the continuous variables were left alone and the data was split into training and testing sets.
        One must separate out these sets to ensure the model is not artifically accurate because it has seen the data before.
        """)
        st.write("Decision Tree Training Set")
        st.dataframe(dt_train)
        st.write("Decision Tree Testing Set") 
        st.dataframe(dt_test)
        st.write("""
        ## Results
        An unrestricted model, where no hyperparameters are set, was used to fit the data. 
                 So when examining the confusion matrix, it should come to no surprise that it has perfect accuracy.
                 The model took as many steps as need to get to pure terminal nodes.
        """)
        st.image("./images/dt_confusion_matrix.png", use_column_width=True, caption='The Confusion Matrix shows that the Decision Tree model is 100% accurate on the training set.')
        st.write('')
        st.image("./images/dt_graph.png", use_column_width=True, caption='It is abundantly clear that while the model can get to some classifications into three steps, there is a very long tail of decisions to finally get to terminal nodes.')
        st.write("""Another attempt was made by removing a couple features, particularly the `month` and the average temperature (`tavg`) to help control some of the colinearity.""")
        st.image("./images/dt2_confusion_matrix.png", use_column_width=True, caption='However even with the removal of features, the model is still overfitting the data.')
        st.image('./images/dt_graph_no_tavg_no_month.png', use_column_width=True, caption='The Decision Tree diagram shows that the model is still overfitting the data but with a greater distribution of steps to terminal nodes.')
        st.write("""One final attempt was made by setting a maximum depth of 5 to the tree 
                 and setting the maximum features to be the square root of the total number of features.
                 This model returned an accuracy of `~0.59` on the testing set. The confusion martix reflects this inaccuracy.""")
        st.image('./images/hyper_dt_confusion_matrix.png', use_column_width=True, caption='The Confusion Matrix shows that the Decision Tree model is 62% accurate on the testing set.')
        st.image("./images/dt_graph_no_tavg.png", use_column_width=True, caption='While hyperparameterizing the model helped reduce the number of terminal nodes, it was at the expense of accuracy.')
        st.write("""
        ## Conclusion
        Similar to the heirarchical clustering method, 
                 the Decision Tree model showcases how there is an intricate relationship between the features 
                 and how many "sub-groupings" are needed to understand how to classify the data. 
                 What is clear however is that the model understands there is a strong relationship between the time of year, the temperature and the energy demand.""")
    if sub_tabs == 'Regression':
        log_train = pd.read_feather("./data/log_x_train.feather")
        log_test = pd.read_feather("./data/log_x_test.feather")
        st.write("""All code can be found in this [notebook](https://github.com/prcedrght/ml-class-renewables/blob/main/data/supervised_learning.ipynb).""")
        st.title("Regression")
        st.write("""
        Typically when discussing Regression, there are two types: Linear and Logisitc. 
                 They both are used to model the relationships between variables and predict outcomes.
                 But they differ in one key way, what type of outcome they are trying to predict. 
                 Linear regression is a method used when attempting to find a 
                 "best fit" of a continuous dependent variable to one or many independent variables while 
                 Logistic regression is used when the dependent variable is binary.
                 Interestingly enough, the statistical method to calculate these relationships are same
                 with the only difference being that Logistic regression uses the Sigmoid function to 
                 the linear inputs in a probability which is used to classify the data. 
        Connected to optimizing logistic regression models is the Maximum Likelihood Estimation (MLE).
        The idea is to optimize the parameters (the independent variables coefficients) 
                 of the model to maximize the likelihood of the observed data.
""")
        st.write("""
        ## Data Prep
        Much like the other modeling methods, the data needed to be prepared for the logistic regression model. 
                 Specifically, the categorical variables were encoded into dummy variables, and the demand lables were coverted into 
                 binary labels deonting either `High` or `Low` demand. It was then split into training and testing sets.""")
        st.write("Logistic Regression Training Set")
        st.dataframe(log_train)
        st.write("Logistic Regression Testing Set")
        st.dataframe(log_test)
        st.write("""
        ## Results
        The Logisitic Regression model performs quite well with an accruarcy of `0.88` on the testing set. 
                 This could be in part because the model is only predicting two classes, `High` and `Low` demand.""")
        st.text("""
Classification Report:
              precision    recall  f1-score   support

           0       0.87      0.89      0.88     13575
           1       0.90      0.88      0.89     14821

    accuracy                           0.89     28396
   macro avg       0.89      0.89      0.89     28396
weighted avg       0.89      0.89      0.89     28396
""")
        st.image('./images/log_confusion_matrix.png', use_column_width=True)
        st.write("""For comparision, a multinomial Naïve Bayes model was also run on the same data 
                 and returned an accuracy of `0.87` on the testing set. These are very similar results, this likely because the features
                 of the model are both approximately independent and the features and target have a linear relationship.""") 
        st.text("""
Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.88      0.86     13575
           1       0.89      0.86      0.87     14821

    accuracy                           0.87     28396
   macro avg       0.87      0.87      0.87     28396
weighted avg       0.87      0.87      0.87     28396
""")
        st.image('./images/mnb_log_confusion_matrix.png', use_column_width=True)

    if sub_tabs == "SVM":
        st.title("Support Vector Machines")
    if sub_tabs == "Ensemble Learning":
        st.title("Ensemble Learning")
with conclusion:
    st.title("Conclusion")
    st.write("Coming Soon!")