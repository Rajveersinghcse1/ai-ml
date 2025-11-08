# Sample E-Commerce Dataset

This folder should contain your e-commerce transaction data.

## Expected Data Format

Your data should have the following columns (or similar):

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| InvoiceNo | Unique transaction identifier | String | INV001234 |
| CustomerID | Unique customer identifier | Integer/String | 12345 |
| InvoiceDate | Date and time of transaction | Datetime | 2024-01-15 10:30:00 |
| ProductID | Product identifier | String | PRD001 |
| Quantity | Number of items purchased | Integer | 2 |
| UnitPrice | Price per unit | Float | 29.99 |
| Country | Customer's country (optional) | String | USA |

## Data Sources

You can use data from various sources:

1. **UCI Machine Learning Repository**
   - Online Retail Dataset
   - URL: https://archive.ics.uci.edu/ml/datasets/online+retail

2. **Kaggle Datasets**
   - E-commerce Data
   - Brazilian E-Commerce Public Dataset by Olist

3. **Your Own Data**
   - Export from your e-commerce platform
   - Database queries
   - Analytics tools

## Loading Your Data

Once you have your data file in this folder, update the notebook:

```python
# In the notebook, replace the sample data generation with:
df = pd.read_csv('data/your_data_file.csv')
```

## Data Privacy

**Important:** Never commit sensitive customer data to version control. Add data files to `.gitignore`:

```
data/*.csv
data/*.xlsx
data/*.json
```
