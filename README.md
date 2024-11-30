# **Decision Tree Model on a Small Dataset**

This repository contains a notebook that demonstrates the process of building a decision tree model from scratch, cleaning and preprocessing a small dataset, visualizing relationships, and evaluating the model's predictions. The project showcases how decision trees can be applied to data while emphasizing the importance of data preprocessing.

---

## **Overview**

- The dataset used for this project was intentionally made small and **dirty**, containing missing values and inconsistent entries to practice cleaning and preprocessing steps.
- Due to the small size of the dataset, the entire dataset was used for training. While this is unconventional, it served as a valuable exercise in understanding decision tree models, their implementation, and their limitations.
- Throughout the project, the cleaning and processing steps significantly reduced the size of the dataset, ultimately leaving only **one row**. This reduction highlights the challenges of handling real-world data and reflects a necessary step in my learning journey.

---

## **Steps Covered in the Notebook**

### **1. Data Loading**
- Loaded a small, dirty dataset to simulate real-world data challenges.

### **2. Data Cleaning**
- **Removed missing values** using `dropna()`. This step ensured that the dataset only contained complete records, but it also caused significant row loss due to the extent of missing data.
- **Converted non-numeric columns** (e.g., `Income` and `Credit_Score`) into numeric using `pd.to_numeric(errors='coerce')`. Invalid entries were replaced with `NaN`, contributing further to row reductions.
- **Encoded categorical columns** (e.g., `Marital_Status`) using `.astype('category').cat.codes`, transforming them into numerical formats suitable for machine learning models.

### **3. Feature and Target Separation**
- Extracted features (`X`) and target variable (`y`) to prepare the dataset for model training:
  - `X`: All columns except the target column (`Defaulted`).
  - `y`: The `Defaulted` column indicating whether a customer defaulted or not.

### **4. Model Training**
- Used the **entire dataset** for training due to its small size:
  ```python
  model.fit(X, y)
  ```
- The absence of a test set means the model was evaluated directly on the training data, which may lead to overfitting and inflated metrics. However, this was done to focus on the mechanics of decision tree models.

### **5. Predictions**
- Predictions were generated for the same dataset used for training:
  ```python
  data['Predicted_Defaulted'] = model.predict(X)
  ```

### **6. Visualizations**
- Created various plots to explore the data and model:
  - A **decision tree diagram** to visualize splits and decision paths.
  - A **heatmap** to understand feature correlations.
  - A **pairplot** to inspect feature distributions and relationships with the target.

---

## **Challenges and Key Learnings**

1. **Dataset Reduction**
   - The dataset reduced to just **one row** after cleaning, primarily due to:
     - Extensive missing values handled by `dropna()`.
     - Invalid data replaced with `NaN` during numeric conversion.
   - This reduction was expected and accepted as a necessary step in handling dirty datasets during the learning phase.

2. **Training on Small Data**
   - Using the entire dataset for training is not ideal for real-world applications but was justified for this project due to the limited dataset size.
   - This approach allowed for hands-on experience in implementing decision trees and interpreting results.

3. **Real-World Data Challenges**
   - Working with dirty data provided insight into the importance of proper data cleaning and preprocessing in machine learning workflows.

---

## **Usage Instructions**

### **Requirements**
- Python 3.7+
- Key Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`

### **Run the Notebook**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/decision-tree-small-dataset.git
   cd decision-tree-small-dataset
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook and execute the cells.

### **Dataset**
- The dataset (`dirty_decision_tree_dataset.csv`) is included in this repository. You can replace it with a dataset of your choice to replicate or extend the project.

---

## **Key Code Snippets**

### Adding Predictions to the Dataset
```python
data['Predicted_Defaulted'] = model.predict(X)
```

### Visualizing the Decision Tree
```python
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=X.columns, class_names=["No Default", "Default"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
```

### Generating a Heatmap
```python
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
```

---

## **Why Publish This Notebook?**

You don't make an omelette without breaking eggs. You also don't become the greatest scorer of all time without bricking some shots. This notebook reflects a crucial part of my learning journey. While the final dataset reduced to just one row, it highlights the **importance of data preprocessing** and **working with small datasets** in the initial stages of mastering machine learning. I hope this serves as a helpful resource for others who are also learning and navigating similar challenges.

---

Feel free to explore, critique, and build upon this project. Contributions and feedback are always welcome!

---
