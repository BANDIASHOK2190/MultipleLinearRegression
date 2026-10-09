# AdPulse — Advertising Sales Analytics

**Prepared by:** Ashok  
**Method:** Multiple Linear Regression  
**Interface:** Streamlit dashboard with a custom midnight-navy / teal / coral visual theme

## Project overview
This project studies the relationship between advertising expenditure on TV, Radio, and Newspaper and the sales recorded in `Advertising.csv`. It keeps the same core machine-learning workflow: inspect the data, select features and target, split the data, train a multiple linear regression model, predict, evaluate, interpret coefficients, and visualize results.

## Files
- `app.py` — interactive dashboard with Overview, Sales Predictor, Model Insights, and Dataset Explorer pages.
- `Advertising.csv` — source dataset.
- `Multiple_Linear_Regression_Ashok.ipynb` — step-by-step notebook version of the analysis.
- `requirements.txt` — required Python packages.

## Run the dashboard
1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Install packages: `python -m pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`
5. Your browser should open the dashboard automatically.

## Run the notebook
Open `Multiple_Linear_Regression_Ashok.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab. Keep `Advertising.csv` in the same folder, then run the cells from top to bottom.

## Notes
- The split uses `test_size=0.20` and `random_state=42` for repeatable results.
- The code reads the actual number of rows from the provided CSV rather than assuming a fixed row count.
- Regression coefficients should be interpreted while holding the other inputs constant. This is an educational model and does not establish causation.
