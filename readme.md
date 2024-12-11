# Should This Loan Be Approved or Denied?

## Introduction
This repository contains the code and resources for predicting whether a small business loan should be approved or denied, based on the historical U.S. Small Business Administration (SBA) dataset "Should This Loan Be Approved or Denied?" (available on Kaggle).

The dataset provides a rich variety of features, including:
- **Loan Terms:** Approved amount, loan term length in months.
- **Borrower Characteristics:** State, city, ZIP code, number of employees.
- **Business Attributes:** Industry classification (NAICS-based), indication of whether the business is new or existing, and franchise information.
- **Loan Programs:** Participation in LowDoc programs, presence of a revolving line of credit (RevLineCr).
- **Outcome (MIS_Status):** Indicates if the loan was Paid in Full (PIF) or Charged Off (CHGOFF).

This dataset presents a complex, real-world scenario with imbalanced classes (approximately 78.4% loans are PIF and 21.6% are CHGOFF), non-linear feature interactions, and varying conditions across geographies and industries.

## Hypothesis and Objectives
**Hypothesis:**  
A tree-based ensemble model (XGBoost) will outperform a linear model (Logistic Regression) in identifying high-risk (CHGOFF) loans due to its ability to handle non-linear relationships, complex feature interactions, and class imbalance more effectively.

**Objectives:**
1. Clean and preprocess the dataset, handling missing values, erroneous entries, and encoding categorical variables.
2. Perform exploratory data analysis (EDA) to understand data distribution, industry and geographic variation, and class imbalance.
3. Compare the performance of Logistic Regression (linear) and XGBoost (non-linear ensemble) in predicting loan outcomes.
4. Validate the hypothesis that XGBoost will provide better recall for charged-off loans while maintaining high overall accuracy.

## Techniques Used
- **Data Cleaning and Feature Engineering:**  
  Dropping irrelevant columns, imputing missing values, converting categorical variables to numeric (one-hot encoding), and engineering features like industry categories from NAICS codes.
  
- **Exploratory Data Analysis (EDA):**  
  Visualizing distributions, checking correlations, and examining patterns by industry, state, and loan attributes.
  
- **Modeling Approaches:**
  - *Logistic Regression:* A baseline linear model for initial comparison.
  - *XGBoost:* A gradient-boosted tree ensemble, better equipped to model complexity and handle imbalance.
  
- **Model Evaluation Metrics:**
  Classification reports, precision, recall, F1-scores, and accuracy. Special emphasis on CHGOFF recall to ensure detection of high-risk loans.

## Results
- **Logistic Regression:**  
  Achieved around 85% accuracy but struggled with the minority CHGOFF class, capturing only about 50% of those loans. It primarily modeled the majority PIF class successfully, leaving defaults underdetected.
  
- **XGBoost:**  
  Achieved about 95% accuracy and significantly improved CHGOFF detection (precision ~0.90, recall ~0.87). This result confirmed the hypothesis that a non-linear, ensemble-based model outperforms a linear model in this scenario.

## Installation and Usage
1. **Clone the Repository:**
 ```bash
 git clone https://github.com/NIU1455751/kagglesba.git
 ```
   
2. **Set Up a Virtual Environment (OPTIONAL)**
Ensure you have Python 3.7+ installed.

  ```bash
  python -m venv venv
  source venv/bin/activate # On Windows: venv\Scripts\activate
  ```

3. **Install Dependencies** 
  ```bash
  cd kagglesba/
  pip install -r requirements.txt
  ```
4.  **Run the Jupyter Notebook**
Start Jupyter Notebook by running:
  ```bash
  jupyter notebook Kaggle.ipynb
  ```
## Results and Outputs
The notebook provides:
- **EDA Insights:** Plots and summaries that explore trends in loan outcomes, geography, and industries.
- **Model Results:** Classification reports for both Logistic Regression and XGBoost models, highlighting accuracy, precision, recall, and F1-scores.
- **Visualizations:** Scatterplots, heatmaps, and other graphics that illustrate the dataset’s characteristics and modeling performance.

## Contact
For questions or additional information, please reach out to me at: marcpomar.cvb@gmail.com


