# CoffeeCoders

The most caffeinated coders of Harris '26

The Coffee Coders are made up of two groups from the University of Chicago Harris School of Public Policy's Data Visualization Course, groups 24 and 49.

**Coffee Coders Group 24:** Dominick Sanakiewicz (dominicksanakiewicz) & Abraham Sadat (abrahamsadat96)

**Coffee Coders Group 49:** Amanda Gu (AmandaAtHarris) & Zhen Zang (UChiZhen)

this is our final project repo, for a full history of commits please see: https://github.com/AmandaAtHarris/CoffeeCoders/commits/main/

## Research Question

This project uses data to explore education outcomes across Cook County using data from 2018-2024.

With the exception of food deserts, which are fixed to 2019, the scope of our data is from 2018 to 2024. Our data was sourced from sources across all levels of government. We used the Census's 5 Year American Community Survey data to capture household income and demographics per census tract. Finally, we used the USDA's Economic Research Survey (ERS) to capture food desert data per census tract. We used a couple of different metrics constructed to model segregation that we encountered in the literature. These include the segregation quotient from Aguirre-Nuñez, Carlos, et al (2024) and the dissimilarity index Green (2022). We used this data as our independent variables.

Our dependent variables came from the Illinois School Board of Education (ISBE) School Report Card. They were: chronic absenteeism, four-year graduation rate, English Language Arts (ELA) proficiency, and math proficiency. The ISBE defines chronic absenteeism as a student being absent for 10 or more percent of the school year, or 18 days. Both ELA and math proficiency are derived from standardized tests.

All of our data is publicly available.

## Data Sources

| Dataset | Source | Level |
|---------|--------|-------|
| American Community Survey (5-Year) | U.S. Census Bureau | Census tract |
| Food Access Research Atlas | USDA ERS (2019) | Census tract |
| School Report Card | Illinois State Board of Education | School |
| School Locations | National Center for Education Statistics | School |
| CTA Bus Routes | Chicago Transit Authority | Route |
| Cook County Shapefiles | U.S. Census Bureau | Census tract |

## Data Processing

The data processing flow is as follows:

1. **`preprocessing.py`** — Cleans all raw data from `data/raw-data/` and merges into `data/derived-data/final_merged.csv`
2. **`ml_pipeline.py`** — Runs the ElasticNet model and outputs coefficients to `outputs/`
3. **`writeup-alphabet.qmd`** — Writeup with static visualizations

## Streamlit Dashboard

**Dashboard link:** https://coffinatedcoders.streamlit.app/

> **Note:** Streamlit Community Cloud apps need to be "woken up" if they have not been run in the last 24 hours. This is normal Streamlit behavior, not a bug.

## Repository Structure

```
CoffeeCoders/
├── data/
│   ├── raw-data/          # Original, unmodified datasets
│   └── derived-data/      # Processed/merged datasets
├── outputs/               # ML model outputs and plots
├── streamlit-app/         # Streamlit dashboard code
├── preprocessing.py       # Data cleaning and merging
├── ml_pipeline.py         # ElasticNet ML pipeline
├── writeup-alphabet.qmd   # Writeup (Quarto)
├── requirements.txt       # Python dependencies
└── .gitignore
```

