import streamlit as st
# from streamlit_option_menu import option_menu
import base64
# import plotly.express as px
import pandas as pd


# Function to read and encode the image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to apply custom CSS
def set_background(png_file, size='cover'):
    bin_str = get_base64_of_bin_file(png_file)
    css_str = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: {size};
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css_str, unsafe_allow_html=True)

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

intro, data, tab3, tab4, tab5 = st.tabs(["Introduction", "Data Gathering", "Data Prep/EDA", "Placeholder 1", "Placeholder 2"])

# Introduction Tab
with intro:
# if selected == "Introduction":
    st.title("Smart Grid Load Prediction with Renewable Engergy Integration & Optimization")
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
    """)

# Data Gathering Tab
with data:
# elif selected == "Data Gathering":
    st.title("Data Gathering")
    # sub_tabs = st.selectbox("Data Gathering Details", ["P1", "P2", "P3"])
    st.write("Content for Data Gathering will go here.")

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

# # Data Prep/EDA Tab
# elif selected == "Data Prep/EDA":
#     st.title("Data Prep/EDA")
#     st.write("Content for Data Preparation and Exploratory Data Analysis will go here.")

# # PCA Tab
# elif selected == "PCA":
#     st.title("PCA")
#     st.write("Content for Placeholder 1 will go here.")

# # ARM Tab
# elif selected == "ARM":
#     st.title("ARM")
#     st.write("Content for Placeholder 2 will go here.")