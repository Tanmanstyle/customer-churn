# Customer Churn Prediction with Scikit-learn

## Project Summary

This project predicts which customers are most likely to leave a fintech or banking service. In business terms, this is called **customer churn prediction**.

The goal is simple:

> Help a business identify high-risk customers early, so the retention team can take action before those customers leave.

The project uses Python and scikit-learn to train a machine learning model that gives each customer:

- a churn probability
- a predicted outcome: likely to stay or likely to churn
- a risk band: Low Risk, Medium Risk, or High Risk

This makes the output useful for both technical users and non-technical business stakeholders.

## Business Problem

Customer churn is when a customer stops using a product or service.

For a fintech or banking company, churn can mean losing revenue, reducing customer lifetime value, and spending more money to replace customers through marketing.

A churn prediction model helps answer questions like:

- Which customers are most likely to leave?
- What customer behaviours are linked with churn?
- Which customers should the retention team contact first?
- Can we turn raw customer data into a practical risk list?

---

## Project Results

In testing, the model achieved a test ROC-AUC of approximately **0.73**.

In plain English:

- `0.50` would mean the model is no better than random guessing
- `1.00` would mean the model is perfect
- `0.73` suggests the model has a useful ability to separate likely churners from likely non-churners

The model is not production-ready, because the data is synthetic, but it is a strong demonstration of the machine learning workflow.

---

## What the Model Produces

After running the project, two main outputs are created:

```text
outputs/churn_predictions.csv
outputs/figures/
```

### `outputs/churn_predictions.csv`

This file contains the model's predictions for customers in the test set.

| Column | Meaning |
|---|---|
| `actual_churn` | The real outcome in the dataset. `0` means stayed, `1` means churned. |
| `predicted_churn_probability` | The model's estimated probability that the customer will churn. |
| `predicted_churn` | The model's final prediction. `0` means predicted to stay, `1` means predicted to churn. |
| `risk_band` | A business-friendly label: Low Risk, Medium Risk, or High Risk. |

Example interpretation:

```text
predicted_churn_probability = 0.82
risk_band = High Risk
```

This means the model estimates that the customer has an 82% chance of churning, so the customer should be prioritised for retention activity.

### `outputs/figures/`

This folder contains charts that explain the data and model performance.

| File | What it shows |
|---|---|
| `01_churn_count.png` | How many customers stayed vs churned. |
| `02_tenure_by_churn.png` | Whether churned customers tend to have shorter or longer tenure. |
| `03_churn_by_complaints.png` | How churn rate changes as complaints increase. |
| `04_correlation_heatmap.png` | Relationships between customer features and churn. |
| `05_confusion_matrix.png` | Where the model predicted correctly and incorrectly. |
| `06_roc_curve.png` | How well the model separates churners from non-churners. |
| `07_precision_recall_curve.png` | The tradeoff between catching churners and avoiding false alarms. |
| `08_feature_coefficients.png` | Which features increase or reduce churn risk. |

---

## Project Structure

```text
customer-churn-sklearn/
├── data/
│   └── fintech_churn_synthetic.csv
├── notebooks/
│   └── customer_churn_sklearn.ipynb
├── outputs/
│   ├── churn_predictions.csv
│   └── figures/
├── src/
│   ├── churn_sklearn.py
│   └── generate_synthetic_churn_data.py
├── requirements.txt
└── README.md
```

---

## Tools Used

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| pandas | Loading and analysing data |
| NumPy | Numerical calculations |
| matplotlib | Creating charts |
| seaborn | Creating statistical visualisations |
| scikit-learn | Building and evaluating the machine learning model |
| Jupyter Notebook | Optional notebook-based project walkthrough |

---

# How to Set Up and Run This Project

This section is written for non-technical users who want to run the project step by step.

## Step 1: Install Python

Install Python 3.10 or newer.

To check whether Python is already installed, open Terminal or Command Prompt and run:

```bash
python --version
```

Depending on your computer, you may need to use:

```bash
python3 --version
```

If Python is installed, you should see something like:

```text
Python 3.10.0
```

or higher.

---

## Step 2: Download or Unzip the Project

If the project is in a zip file, unzip it first.

You should end up with a folder called:

```text
customer-churn-sklearn
```

Open that folder. You should see files and folders such as:

```text
data
src
outputs
requirements.txt
README.md
```

---

## Step 3: Open a Terminal in the Project Folder

You need to run the commands from inside the project folder.

### On Mac

Open Terminal, then move into the project folder. Example:

```bash
cd Downloads/customer-churn-sklearn
```

### On Windows

Open Command Prompt or PowerShell, then move into the project folder. Example:

```powershell
cd Downloads\customer-churn-sklearn
```

You can confirm you are in the right place by running:

```bash
ls
```

On Windows Command Prompt, use:

```cmd
dir
```

You should see `requirements.txt` and the `src` folder.

---

## Step 4: Create a Virtual Environment

A virtual environment keeps this project's Python packages separate from everything else on your computer.

Run:

```bash
python -m venv .venv
```

If that does not work on Mac, try:

```bash
python3 -m venv .venv
```

---

## Step 5: Activate the Virtual Environment

### Mac or Linux

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

When it works, you should see `(.venv)` at the start of your terminal line.

---

## Step 6: Install the Required Packages

Run:

```bash
pip install -r requirements.txt
```

This installs the Python libraries used by the project, including pandas, seaborn, matplotlib, and scikit-learn.

---

## Step 7: Run the Project

Run:

```bash
python src/churn_sklearn.py
```

If you are on Mac and `python` does not work, try:

```bash
python3 src/churn_sklearn.py
```

The script will:

1. Load the customer churn dataset
2. Create exploratory charts
3. Train the machine learning model
4. Evaluate the model
5. Save prediction results
6. Save model performance charts

---

## Step 8: View the Results

After the script finishes, open:

```text
outputs/churn_predictions.csv
```

You can open this file in Excel, Google Sheets, Numbers, or any spreadsheet tool.

Also open:

```text
outputs/figures/
```

This folder contains the charts created by the project.

### Mac

```bash
open outputs/figures
open outputs/churn_predictions.csv
```

### Windows

```powershell
start outputs\figures
start outputs\churn_predictions.csv
```

---

## Optional: Open the Jupyter Notebook

The notebook gives a more visual walkthrough of the project.

First install Jupyter if it is not already installed:

```bash
pip install notebook
```

Then run:

```bash
jupyter notebook notebooks/customer_churn_sklearn.ipynb
```

A browser window should open with the notebook.

---

## Optional: Regenerate the Synthetic Dataset

The project already includes a dataset, but you can recreate it by running:

```bash
python src/generate_synthetic_churn_data.py
```

Then run the main project again:

```bash
python src/churn_sklearn.py
```

---

## Common Setup Problems

### Problem: `python` is not recognised

Try:

```bash
python3 --version
```

If that works, use `python3` instead of `python` in the commands.

---

### Problem: `pip` is not recognised

Try:

```bash
python -m pip install -r requirements.txt
```

or:

```bash
python3 -m pip install -r requirements.txt
```

---

### Problem: The dataset cannot be found

Make sure you are running the script from the main project folder, not from inside the `src` folder.

Correct:

```bash
python src/churn_sklearn.py
```

Incorrect:

```bash
cd src
python churn_sklearn.py
```

---

### Problem: Seaborn gives `Invalid RGBA argument: None`

Some seaborn/matplotlib versions can raise this charting error.

If it happens, open:

```text
src/churn_sklearn.py
```

Find the barplot line:

```python
sns.barplot(data=df, x="complaints", y=TARGET, estimator=np.mean)
```

Replace it with:

```python
sns.barplot(
    data=df,
    x="complaints",
    y=TARGET,
    estimator=np.mean,
    color="steelblue",
)
```

Then run the project again.

---

# How the Model Works

The model uses logistic regression, a common and explainable machine learning method for yes/no predictions.

In this project, the yes/no question is:

> Will this customer churn?

The model looks at customer features such as:

- age
- tenure
- account balance
- credit score
- number of products
- active status
- estimated salary
- complaints
- support tickets
- mobile logins
- overdraft count

It then estimates the probability that each customer will churn.

---

## Why the Project Uses a Pipeline

The project uses a scikit-learn `Pipeline` to combine preprocessing and modelling.

The pipeline does two things:

1. Standardises the numeric features with `StandardScaler`
2. Trains a `LogisticRegression` model

This is a good practice because it keeps the workflow clean and helps prevent data leakage.

---

## Why Class Imbalance Matters

In churn problems, most customers usually stay and only a smaller percentage churn.

That means a model could appear accurate by simply predicting that everyone stays.

For example, if only 10% of customers churn, a model could predict “stayed” for every customer and still be 90% accurate. But that model would be useless because it would miss every churned customer.

This project handles that issue with:

```python
class_weight="balanced"
```

This tells the model to pay more attention to the smaller churn group.

---

## Why Threshold Optimisation Matters

Many models use a default 50% probability cutoff.

For example:

- 49% churn probability = predicted to stay
- 51% churn probability = predicted to churn

That is not always the best choice.

This project uses the precision-recall curve to find a better threshold for balancing:

- catching customers who may churn
- avoiding too many false alarms

---

## Business Interpretation

The final output can be used by a retention team.

A simple business workflow could be:

1. Run the model on recent customer data
2. Sort customers by churn probability
3. Focus first on High Risk customers
4. Review the reasons suggested by feature patterns
5. Offer support, discounts, onboarding help, or other retention actions
6. Track whether the intervention reduces churn

The model does not make the final business decision. It helps prioritise attention.

---

## Limitations

This project uses synthetic data. That means the data was generated for demonstration purposes and does not represent real customers.

Because of that, the results should be interpreted as a demonstration of technical skill and machine learning workflow, not as real business findings.

Before using a churn model in a real company, the following would be needed:

- real customer data
- data quality checks
- privacy and compliance review
- fairness and bias checks
- model monitoring
- stakeholder approval
- testing against real business outcomes

---

## Future Improvements

Possible next steps:

- Compare logistic regression with Random Forest, XGBoost, or Gradient Boosting
- Add hyperparameter tuning with `GridSearchCV`
- Add SHAP values for better explainability
- Create a dashboard for business users
- Add automated tests with `pytest`
- Export a ranked list of high-risk customers for a retention campaign
- Track model performance over time
