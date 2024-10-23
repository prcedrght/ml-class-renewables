import streamlit as st
# from streamlit_option_menu import option_menu
import base64
# import plotly.express as px
import pandas as pd

# national_capacity = pd.read_csv("./data/national_generation_capacity_stacked_filtered.csv")
# national_capacity_cleaned = national_capacity[['technology', 'year', 'country', 'capacity']].copy()
# time_data = pd.read_csv("./data/time_series_60min_singleindex_filtered.csv")
# long_time_data = pd.read_pickle("./data/long_pickle.pkl")


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
        In order to have a solid description of energy habits, a wide variety of data must be collected: annual consumption measurements for long term trends, time series consumption patterns, and weather history. 
        While there are several different renewable energy sources that should be considered for this investigation, 
        the sparsity of certain types like geothermal means this data is primarily made up of photovolatic (solar) and wind energy. 
        The data used in this project was collected from the following sources: the [Open Power System Data (OPSD) project](https://open-power-system-data.org/), 
        and [Renewable.ninjas](https://www.renewables.ninja/). \n \n
        ##
        ##### Open Power System Data (OPSD) Project
        The OPSD project contains data about energy generation capacities and consumption for several European countries. 
        The project collects, processes and documents publicly available data on various technologies, individual power plants as well as varying aggregated time series data for countries.
        The data is availble in a couple different formats, but for the purposes of this project, the data was downloaded in CSV format. 
        Specifically, the data used in this project were: the [Time series data](https://doi.org/10.25832/time_series/2020-10-06) which contains hourly data on electricity generation and consumption for several European countries, 
        and the [National generation capacity](https://doi.org/10.25832/national_generation_capacity/2020-10-01) which has measurements for annual country consumption by technology types. \n
        ##
        ##### Renewables.ninja
        Renewables.ninja is an API that provides solar and wind power models for researching the effects of integrating renewable technologies into smart grids.
        The data is accessible via JSON formats by querying two different endpoints: solar (htttps://www.renewables.ninja/api/data/pv) and wind (https://www.renewables.ninja/api/data/wind).
        By feeding the API latitude and longitude coordinates, along with the desired year and a few other parameters, the response will return hourly data of simulated electrical generation by the respective technologies along with weather information.
        For example: ```curl -H 'Authorization: Token <your_token_here>' -X GET 'https://www.renewables.ninja/api/data/wind?&lat=56&lon=-3&date_from=2014-01-01&date_to=2014-02-28&capacity=1&dataset=merra2&height=100&turbine=Vestas+V80+2000&format=json'```
        """)
    # st.write("Content for Data Gathering will go here.")

    if sub_tabs == "Data Cleaning":
        st.write("""
        #### DATA CLEANING
        One of the data sets used in this project is an annual summary of energy capacity by country.
        """)
        # st.dataframe(national_capacity.iloc[0:5])
        st.write("""However, there are several dimensions that are not required from this data set and thus removed.""")
        # st.dataframe(national_capacity_cleaned.iloc[0:5])
        st.write("""
        Another data set from the OPSD project is time series information that is in a wide format. Columns all share similar suffixes, e.g. `DE_soloar_profile` and `AT_solar_profile`, but the prefixes are the country codes. This data needs to be transformed into long format and have null records dropped.
        """)
        # st.dataframe(time_data.iloc[0:5])
        # st.dataframe(long_time_data)


    if sub_tabs == "Exploratory Data Analysis":
        st.write("""
        #### EXPLORATORY DATA ANALYSIS

        """)
        # st.image("./images/total_annual_capacity_by_country.png", use_column_width=True, caption='Total Annual Capacity')
        # st.image("./images/annual_capacity_by_country_and_technology.png", use_column_width=True, caption='Annual Capacity by Country & Technology')
        # st.image("./images/pv_hist.png", use_column_width=True, caption='Photovoltaic Histogram')
        # st.image("./images/pv_country_box.png", use_column_width=True, caption='Photovoltaic Boxplot by Country')
        # st.image("./images/wind_hist.png", use_column_width=True, caption='Wind Histogram')
        # st.image("./images/wind_country_box.png", use_column_width=True, caption='Wind Boxplot by Country')

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
    st.title("Modeling")
    st.write("Coming Soon!")

with conclusion:
    st.title("Conclusion")
    st.write("Coming Soon!")