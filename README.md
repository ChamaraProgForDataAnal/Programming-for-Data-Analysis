# Nutritional Food Analysis

This repository contains the analysis of a nutritional dataset, focusing on identifying patterns, exploring subsets of foods, and visualizing relationships between nutrients. The analysis is designed for insights into dietary goals such as high-protein, low-carb, or high-fat, low-sugar foods.

---

## **Objective**
To analyze and visualize a nutritional dataset to uncover patterns, identify representative food groups, and provide actionable insights for dietary planning.

---

## **Steps in Analysis**

### **1. Data Preprocessing**
- Loaded and inspected the dataset for structure and missing values.
- Handled missing values by imputing medians for numeric columns.
- Standardized numeric data for uniform scaling during analysis.

### **2. Exploratory Data Analysis (EDA)**
- Visualized distributions of key nutrients such as:
  - Calories (`Energ_Kcal`).
  - Protein (`Protein_(g)`).
  - Lipids (`Lipid_Tot_(g)`).
  - Carbohydrates (`Carbohydrt_(g)`).
- Generated pairwise scatter plots and correlation heatmaps to explore relationships between nutrients.

### **3. Subset Analysis**
#### **High-Protein, Low-Carb Foods**
- Filtered foods in the top 25% of protein and bottom 25% of carbohydrates.
- Identified unique characteristics:
  - High protein (~26g on average).
  - Minimal carbs, sugar, and fiber.
- Visualized nutrient distributions using bar and radar charts.

#### **High-Fat, Low-Sugar Foods**
- Filtered foods in the top 40% of fats and bottom 40% of sugars.
- Observed characteristics:
  - High fats (~25g on average).
  - Moderate protein levels (~19.5g).
  - Minimal sugar and carbs.

### **4. Dimensionality Reduction and Clustering**
- Applied **PCA** to reduce the dimensions of the dataset for visualization.
- Performed **K-Means Clustering**:
  - Created 4 clusters representing groups of foods with similar nutrient profiles.
  - Visualized clusters using PCA scatter plots.

### **5. Representative Foods**
- Identified representative foods for each cluster based on proximity to cluster centroids.
- Representative foods showcase unique nutrient compositions for each group.

---

## **Visualizations**
- **Histograms**: Showed nutrient distributions.
- **Radar Charts**: Highlighted nutritional profiles of specific subsets (e.g., high-protein, low-carb).
- **Scatter Plots**: Explored pairwise relationships and PCA-based clusters.

---

## **Insights**
1. High-protein, low-carb foods are dominated by lean meats and seafood.
2. High-fat, low-sugar foods feature oils and high-calorie items.
3. Nutritional clustering reveals distinct food groups with unique calorie and nutrient distributions.

---

## **Usage**
- **Dietary Recommendations**:
  - For high-protein, low-carb diets: Lean meats, seafood, eggs, and tofu.
  - For high-fat, low-sugar diets: Oils like safflower and palm.
- **Clustering for Meal Planning**:
  - Use cluster insights to plan balanced meals based on nutrient preferences.

---

## **Future Work**
1. Explore interactions between micronutrients (e.g., vitamins and minerals).
2. Build interactive dashboards for better exploration.
3. Incorporate advanced machine learning for automated dietary recommendations.

---

## **Requirements**
- Python 3.8+
- Libraries:
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `sklearn`
