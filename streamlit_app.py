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
    st.write("""Earth is in a crisis. Since the dawn of the industrial revolution modern society has become heavily dependent on the finite resources of fossil fuels which are a major contributor, if not the sole driver, behind modern [warming trends](https://bpb-eu-w2.wpmucdn.com/blogs.reading.ac.uk/dist/3/187/files/2020/01/lia_mwp-1.png). This existential threat is true for every single person on Earth, no one will be able to totally avoid the turmoil this is causing. Heat waves, droughts, deep freezes, flooding, famine and diseases will all be inescapable. While [experts suggest](https://fortune.com/2024/04/10/united-nations-climate-chief-humanity-has-2-years-to-save-the-world/) there is time to curb the effects of climate change, there is still significant effort needed to ensure a future for the next generations. Thankfully a bright spot can be seen amongst the smog — the advancement of “green” energy sources and technologies. The ability to harness solar, wind, hydropower, or geothermal energy unlocks a paradigm that is rooted in clean, renewable power generation, and one that mitigates humanity’s impact on the planet. These sources of energy are exciting but they come with shared challenges. For example hydropower — like energy that comes from a dam or newer technology that harnesses the [kinetic energy of the tides and ocean currents](https://www.pnnl.gov/explainer-articles/tidal-energy#:~:text=What%20is%20tidal%20energy%3F,the%20water%20to%20move%20faster.) — requires significant infrastructure to move the energy captured from its point of origin to the consumer. As of 2023, the Hoover Dam was capable of generating [1,080 megawatts](https://www.statista.com/statistics/1360603/power-capacity-of-the-hoover-dam/) at any given moment. That is enough energy to power for roughly one million homes. However the dam is located in a relatively remote location (as are most dams), and Los Angeles which is the city that [consumes most of that energy (15%)](https://www.eia.gov/kids/for-teachers/field-trips/hoover-dam-hydroelectric-plant.php#:~:text=Nineteen%20percent%20of%20the%20electricity,that%20is%20270%20miles%20away.) is 270 miles away. Aside from proximity challenges, each renewable energy source faces the challenge of relying on non-human factors like the weather — solar energy requires sunshine, wind requires wind, etc. This means that renewable energy sources have a unique set of optimal weather conditions in which they can operate most efficiently.""")
    st.write("""With the invention of the internet and interconnectedness of things, also came the ability to better understand and monitor energy consumption on the electric grid. This technology gave birth to the smart grid. A [smart grid relies on two-way communication](https://ieeexplore.ieee.org/document/8452969) between the power stations, batteries and the consumers to better utilize the energy available. For example, if energy is in high demand, and customers are connected to a smart grid, energy companies can adjust thermostat settings to manage consumption and ensure blackouts do not occur. There is complicated modeling involved in balancing the supply and demand, and renewable energy sources' notoriety for being sporadic make that more difficult. By examining the relationship between renewable energy efficiency and grid demand, new evidence could be used in modeling to better plan for grid stress and ease for storage.""")
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
        st.write("""As a final preprocessing step, the date column was encoded into two `sin` and `cos` cyclical features.
                 This helps ensure the models know that December is closer to January than it is to June.""")
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
    sub_tabs = st.selectbox("Use dropdown to explore different models:", ["PCA", "Clustering", "ARM"])
    if sub_tabs == "PCA":
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
        st.write("""Becuase the data contains dimension that are not quantitative, those must be removed before trasnforming it. The transformation returns an array of values.""")
        st.dataframe(pca_head)
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
        st.write("""While this is a good start, in order to get to 95% of the explained variance, `n=7` components are required.""")
        st.image('./images/cumulative_variance.png', use_column_width=True, caption='The cumulative variance plot shows that 95% of the variance is explained by 7 components.')
        st.write("""The top three eigenvalues of this data are:""")
        st.text('[2.69345609 1.73470563 1.4381734]')
    
    if sub_tabs == "Clustering":
        st.title("Clustering")
        st.write("""
        The data preparation for clustering is the same as prepping for PCA using `n=3` components.
        ### KMeans
        First to determine how many clusters are appropriate, the Silhouette Scores were calculated for a range of cluster sizes.""")
        st.code("""k_values = range(2, 11)
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pc_3)
    score = silhouette_score(pc_3, kmeans.labels_)
    silhouette_scores.append(score)""")
        

with conclusion:
    st.title("Conclusion")
    st.write("Coming Soon!")