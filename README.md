# **A Machine Learning-Driven Feasibility Assessment Tool for Converting Abandoned Oil & Gas Wells into Geothermal Energy Production Facilities**

---
[Streamlit Link](https://geothermaloilgaswellrepclassification.streamlit.app/)

## Project Overview

This repository contains a complete end-to-end data science project that develops and deploys a machine learning classification model to predict the feasibility of converting abandoned oil and gas wells into geothermal energy production assets.

### Key Objectives:
- **Predict conversion success** using 10 critical well parameters
- **Enable evidence-based portfolio screening** for PLN (Indonesian state power company)
- **Reduce capital deployment risk** by identifying viable candidates in well portfolios
- **Support Indonesia's renewable energy mandate** (100% by 2050)

### Business Context:
- **Problem**: Indonesia has 3,000+ legacy oil & gas wells; only 37.5% of conversion projects succeed historically
- **Solution**: ML model predicts success with **85.7% accuracy** and **0.79 ROC-AUC**
- **Impact**: USD 20-50M NPV potential across a 10-project portfolio, plus 50-150 MW renewable capacity

---

## Dataset

**Source**: 24 international abandoned well conversion projects (1987-2025)
- **Countries**: USA (9), Europe (6), Asia-Pacific (6), South America (2), Other (1)
- **Technologies**: Open-loop, Closed-loop BHE, EGS, Binary cycle, Hybrid
- **Success Rate**: 37.5% (9 successes, 15 failures)

**Target Variable**: Binary classification
- `1` = Successful conversion
- `0` = Failed conversion

---

## 🔧 Features (10 Numerical Parameters)

| Feature | Unit | Range | Description |
|---------|------|-------|-------------|
| `Well_Depth_m` | meters | 876–5,000 | Total drilling depth |
| `Bottomhole_Temp_C` | °C | 85–220 | Temperature at well bottom |
| `Flow_Rate_m3_day` | m³/day | 0–6,350 | Production flow capacity |
| `Flow_Rate_Stability_pct` | % | 0–100 | Consistency of production rate |
| `Power_Output_kW` | kW | 0–11,400 | Expected electrical capacity |
| `Well_Age_Years` | years | 5–65 | Age of the well |
| `Distance_to_Demand_km` | km | 5–150 | Distance to market/users |
| `Permeability_mD` | millidarcies | 0.2–100 | Rock permeability coefficient |
| `TDS_mg_L` | mg/L | 0–10,000 | Total dissolved solids |
| `H2S_ppm` | ppm | 5–800 | Hydrogen sulfide concentration |

---

##  Repository Structure

```
Geothermal_OilGas_Well_Rep_Classification/
│
├── README.md                                  # Project documentation
├── requirement.txt                            # Python dependencies
│
├── app.py                                     # Streamlit web application
├── final_model.sav                            # Trained Gradient Boost
│
├── end-to-end-geothermal-classification.ipynb # Full analysis notebook
├── geothermal_classification_data.csv         # Raw dataset
│
├── Documentation.pdf                          # Detailed technical report
└── README                                     # Repository summary
```

##  Application Usage

### **Single Well Prediction**

1. **Enter Parameters** (3 columns):
   - **Thermal & Depth**: Temperature, Well Depth, Permeability
   - **Fluid Dynamics**: Flow Rate, Flow Rate Stability, TDS, H2S
   - **Operational**: Power Output, Well Age, Distance to Demand

2. **Click "🔮 Predict Feasibility"**

3. **View Results**:
   -  Prediction: Viable or High Risk
   -  Success Probability (0-100%)
   -  Interactive gauge chart
   -  Input data summary

### **Example Inputs (High Success Probability)**
```
Temperature:        155°C
Flow Rate:          3,000 m³/day
Flow Rate Stability: 35%
Well Depth:         2,800 m
Power Output:       1,500 kW
```

---

##  Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 83.3% |
| **Precision** | 83.3% |
| **Recall** | 90% |
| **F1-Score** | 0.826 |
| **ROC-AUC** | 0.9 |
| **CV F1-Mean** | 0.893 |

### Model Selection:
- **Algorithm**: Gradient Boosting Classifier (with scikit-learn ColumnTransformer pipeline)
- **Hyperparameters**: 50 estimators, max depth=3
- **Validation**: 5-fold stratified cross-validation

### Key Predictive Features:
<img width="496" height="290" alt="image" src="https://github.com/user-attachments/assets/204ac484-79a8-474e-a0b2-fb685f9c2528" />

### **end-to-end-geothermal-classification.ipynb**
Complete Jupyter notebook covering:
- Data loading and exploration
- Feature engineering and preprocessing
- Model training (8 algorithms compared)
- Cross-validation and performance evaluation
- Feature importance analysis
- Production model export

## 🛠️ Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
pickles
plotly>=5.14.0
xgboost>=1.7.0
scikit-learn>=1.3.0
```

## 📝 Model Limitations

⚠️ **Sample Size**: Dataset of 24 projects (8% of global repurposing efforts) — elevated uncertainty

⚠️ **Geographic Bias**: Overrepresents USA/Europe mature basins; extrapolation risk for SE Asia

⚠️ **Temporal Bias**: Long-term (15-25 year) viability not fully documented

⚠️ **Class Imbalance**: 37.5% success rate; model exhibits conservative bias toward failure prediction

---

## 📚 References

1. Ashena, S., et al. (2023). "Feasibility analysis of converting abandoned oil and gas wells into geothermal wells and power generation." *Energy*, 265, 126-135.

2. Penn State & Stanford University (2022-2025). "Repurposing abandoned wells for geothermal energy." *Renewable Energy*, multiple publications.

3. International Energy Agency (2024). "Renewable Energy Integration: Global Progress Report."

4. Indonesian Ministry of Energy (2023). "Indonesia Energy Outlook 2023: Pathway to 100% Renewable by 2050."

## 👨‍💻 Author

**Yonathan Hary Hutagalung**  
Data Scientist | Sustainability & Energy Analytics  
GitHub: [YonathanHH](https://github.com/YonathanHH)
