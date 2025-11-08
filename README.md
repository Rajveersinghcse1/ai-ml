# 🎯 Customer Segmentation Dashboard

> **Streamlit-powered** interactive dashboard for customer segmentation analysis

## ✨ Features

- 🚀 **Auto-executing analysis** - Runs on startup
- 📊 **Interactive UI** - Beautiful Streamlit interface
- 🎯 **5 Dashboard Pages**:
  - 🏠 Overview
  - 📈 RFM Analysis
  - 🎯 Segmentation
  - 🤖 ML Clustering
  - 💡 Business Insights
- 📥 **Export Data** - Download CSV files
- 🔄 **Real-time Refresh** - Update analysis instantly

## 🚀 Quick Start

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Start Dashboard

**Option A: Double-click**
```
start_streamlit.bat
```

**Option B: Command line**
```bash
streamlit run streamlit_app.py
```

### 3️⃣ Access Dashboard
```
Automatically opens in browser
Or go to: http://localhost:8501
```

## 📁 Project Structure

```
project/
├── streamlit_app.py                    ⭐ Main Streamlit app
├── start_streamlit.bat                 🚀 Quick launcher
├── requirements.txt                    📦 Dependencies
├── customer_segmentation_analysis.ipynb  📓 Jupyter notebook
├── src/                                📂 Analysis modules
│   ├── preprocessing.py                  - Data cleaning
│   ├── rfm_analysis.py                   - RFM calculations
│   └── clustering.py                     - ML clustering
└── data/                               📂 Data files
    └── Online Retail.xlsx                - Sample dataset
```

## 🎨 Dashboard Pages

### 🏠 Overview
- 6 key metrics cards
- Data processing summary
- Quick insights

### 📈 RFM Analysis
- RFM statistics (mean, median, range)
- Distribution charts
- Score visualizations
- Data explorer

### 🎯 Segmentation
- Segment distribution (bar + pie charts)
- Detailed segment table
- Top 3 segment insights
- Export functionality

### 🤖 ML Clustering
- Clustering quality metrics
- Cluster analysis charts
- Detailed cluster characteristics
- Export functionality

### 💡 Business Insights
- Top performing segments
- At-risk customer alerts
- Growth opportunities
- Strategic recommendations
- Key performance indicators

## 📊 What It Does

1. **Loads Data** - Automatically from `data/Online Retail.xlsx`
2. **Cleans Data** - Removes invalid records
3. **Calculates RFM** - Recency, Frequency, Monetary
4. **Segments Customers** - 10 business segments
5. **ML Clustering** - K-Means (4 clusters)
6. **Generates Insights** - Actionable recommendations
7. **Displays Results** - Interactive visualizations

## 🔄 How It Works

### Auto-Execution
- Analysis runs when app starts
- Results cached for performance
- Click "Refresh Analysis" to update

### Navigation
- Use sidebar to switch pages
- Each page shows different analysis
- All data interconnected

### Export
- Download CSV files from any page
- Includes RFM data, segments, clusters
- Ready for further analysis

## 📦 Dependencies

- **streamlit** - Web framework
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib** - Visualizations
- **seaborn** - Statistical plots
- **scikit-learn** - Machine learning
- **openpyxl** - Excel file reading

## 🎯 Key Benefits

### ✅ Simple
- One command to start
- No API needed
- No frontend files

### ✅ Interactive
- Real-time filtering
- Expandable sections
- Downloadable data

### ✅ Complete
- All notebook outputs
- Multiple visualizations
- Business insights

### ✅ Professional
- Beautiful UI/UX
- Responsive design
- Easy navigation

## 🛠️ Customization

### Change Data Source
Edit line 120 in `streamlit_app.py`:
```python
data_path = 'data/Online Retail.xlsx'  # Change this
```

### Adjust Number of Clusters
Edit line 135 in `streamlit_app.py`:
```python
kmeans_result = perform_kmeans_clustering(rfm, n_clusters=4)  # Change 4
```

### Modify Theme
Edit the CSS in `st.markdown()` at the top of `streamlit_app.py`

## 📈 Performance

- **Startup Time:** ~10 seconds (analysis execution)
- **Page Load:** <1 second (cached results)
- **Memory Usage:** ~50MB
- **Data Size:** 540K+ rows processed

## 🔍 Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Data file not found
**Solution:** Ensure `data/Online Retail.xlsx` exists

### Issue: Port already in use
**Solution:** Kill existing process or use different port
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Issue: Slow startup
**Normal:** First run takes 10-15 seconds for analysis
**If persists:** Check data file size

## 💡 Tips

1. **Use sidebar** for quick navigation
2. **Expand sections** to see more details
3. **Download data** for offline analysis
4. **Click Refresh** to re-run analysis
5. **Share URL** with team members

## 🎉 Advantages Over Previous System

| Feature | Flask | Streamlit |
|---------|-------|-----------|
| Setup | Complex | Simple |
| UI/UX | Manual HTML/CSS | Built-in |
| Interactivity | Limited | Native |
| Updates | Page refresh | Real-time |
| Code | Split files | Single file |
| Learning curve | High | Low |

## 📞 Quick Commands

### Start App
```bash
streamlit run streamlit_app.py
```

### Install Packages
```bash
pip install -r requirements.txt
```

### Stop Server
```
Ctrl+C in terminal
```

### Change Port
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Open in Browser
```bash
streamlit run streamlit_app.py --server.headless false
```

## ✅ Success Checklist

After setup:
- ✅ Dependencies installed
- ✅ Data file exists
- ✅ App starts without errors
- ✅ Browser opens automatically
- ✅ Dashboard displays data
- ✅ Can navigate between pages
- ✅ Charts render correctly
- ✅ Can download CSV files

## 🎊 Summary

**This is the FINAL version:**
- ✅ Single Python file (streamlit_app.py)
- ✅ No Flask/API needed
- ✅ No HTML/CSS/JS files
- ✅ Beautiful interactive UI
- ✅ Auto-executes analysis
- ✅ 5 comprehensive pages
- ✅ Export functionality
- ✅ Professional design

**Just run:** `streamlit run streamlit_app.py`

**That's it!** 🚀

---

**Version:** 4.0 - Streamlit Edition  
**Date:** October 16, 2025  
**Tech Stack:** Streamlit + Python  
**Pages:** 5 interactive dashboards  
**Lines of Code:** ~1,100 (streamlit_app.py)
