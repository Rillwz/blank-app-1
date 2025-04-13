# pip install streamlit pandas numpy scikit-learn matplotlib seaborn

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
@st.cache_data
def load_data():
    data = {
        'Age': [17, 21, 20, 18, 19, 21, 27, 21, 20, 20, 20],
        'Gender': ['Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Female', 'Female'],
        'Weight': [60, 70, 78, 60, 58, 50, 54, 55, 41, 78, 78],
        'Height': [170, 170, 160, 162, 160, 180, 155, 172, 157, 160, 160],
        'Fruit_Consumption': [3, 1, 3, 1, 3, 1, 5, 3, 3, 3, 3],
        'Vegetable_Consumption': [5, 2, 2, 4, 3, 1, 4, 3, 3, 2, 2],
        'Water_Intake': [3, 3, 5, 2.5, 7, 10, 8, 7, 8, 7, 7],
        'Stress_Level': [1, 10, 7, 7, 5, 1, 5, 7, 7, 8, 8]
    }
    df = pd.DataFrame(data)
    
    # Calculate Nutrition Score (new feature)
    scaler = MinMaxScaler()
    nutrition_components = df[['Fruit_Consumption', 'Vegetable_Consumption', 'Water_Intake']]
    scaled_components = scaler.fit_transform(nutrition_components)
    df['Nutrition_Score'] = scaled_components.mean(axis=1) * 10  # Scale to 0-10
    
    # Calculate BMI (additional health metric)
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    
    return df

df = load_data()

# Convert categorical variables to numerical
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Initialize session state variables
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'features_used' not in st.session_state:
    st.session_state.features_used = []
if 'target_var' not in st.session_state:
    st.session_state.target_var = None

# Streamlit app
st.title("Health & Nutrition Analysis Dashboard")

# Sidebar for user input
with st.sidebar:
    st.header("User Input Form")
    with st.form("user_input"):
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=60)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        fruit = st.slider("Fruit Consumption (1-5 scale)", 1, 5, 3)
        vegetable = st.slider("Vegetable Consumption (1-5 scale)", 1, 5, 3)
        water = st.number_input("Water Intake (glasses/day)", min_value=1, max_value=20, value=5)
        stress = st.slider("Stress Level (1-10 scale)", 1, 10, 5)
        
        submitted = st.form_submit_button("Submit and Predict")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Data Exploration", "Nutrition Analysis", "Regression Model", "Prediction"])

with tab1:
    st.header("Dataset Overview")
    st.write(df)
    
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    # Distribution plots for numerical columns
    st.subheader("Distribution of Variables")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    selected_col = st.selectbox("Select a variable to view distribution", num_cols)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[selected_col], bins=10, edgecolor='black')
    ax.set_title(f'Distribution of {selected_col}')
    ax.set_xlabel(selected_col)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    # Boxplot by Gender
    st.subheader("Boxplot by Gender")
    selected_num_col = st.selectbox("Select a numerical variable", [col for col in num_cols if col != 'Gender'])
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    gender_groups = [df[df['Gender'] == 0][selected_num_col], df[df['Gender'] == 1][selected_num_col]]
    ax2.boxplot(gender_groups, labels=['Female', 'Male'])
    ax2.set_title(f'Boxplot of {selected_num_col} by Gender')
    ax2.set_ylabel(selected_num_col)
    st.pyplot(fig2)
    
    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr_matrix = df.corr()
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
    st.pyplot(fig3)

with tab2:
    st.header("Nutrition Analysis")
    
    # Nutrition Score Explanation
    st.markdown("""
    ### Nutrition Score Calculation
    The nutrition score is calculated based on:
    - Fruit Consumption (1-5 scale)
    - Vegetable Consumption (1-5 scale)
    - Water Intake (glasses/day)
    
    All components are normalized and combined into a 0-10 score.
    """)
    
    # Nutrition components visualization
    st.subheader("Nutrition Components Analysis")
    
    # Radar chart for nutrition components
    def plot_radar_chart(row):
        categories = ['Fruits', 'Vegetables', 'Water']
        values = row[['Fruit_Consumption', 'Vegetable_Consumption', 'Water_Intake']].values
        values = np.append(values, values[0])  # Close the radar chart
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='skyblue', alpha=0.6)
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(f"Nutrition Profile (Person {row.name})", size=14)
        ax.set_rlabel_position(30)
        plt.yticks([1, 3, 5, 7, 9], ["1", "3", "5", "7", "9"], color="grey", size=8)
        plt.ylim(0,10)
        return fig
    
    selected_person = st.selectbox("Select person to view nutrition profile", df.index)
    st.pyplot(plot_radar_chart(df.loc[selected_person]))
    
    # Nutrition vs Stress
    st.subheader("Nutrition vs Stress Levels")
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='Nutrition_Score', y='Stress_Level', hue='Gender', 
                   palette={0: 'pink', 1: 'blue'}, s=100)
    ax5.set_title("Nutrition Score vs Stress Level")
    ax5.set_xlabel("Nutrition Score (0-10)")
    ax5.set_ylabel("Stress Level (1-10)")
    st.pyplot(fig5)
    
    # BMI vs Nutrition
    st.subheader("BMI vs Nutrition Score")
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    sns.regplot(data=df, x='Nutrition_Score', y='BMI', scatter_kws={'s': 80})
    ax6.set_title("BMI vs Nutrition Score")
    ax6.set_xlabel("Nutrition Score (0-10)")
    ax6.set_ylabel("BMI")
    st.pyplot(fig6)

with tab3:
    st.header("Linear Regression Model")
    
    # Let user select target and features
    available_features = [col for col in df.columns if col != 'Gender']
    target = st.selectbox("Select target variable", available_features)
    features = st.multiselect("Select features", 
                             [col for col in available_features if col != target],
                             default=[col for col in available_features if col not in [target, 'Gender']])
    
    if features and target:
        # Prepare data
        X = df[features]
        y = df[target]
        
        # Convert categorical variables to dummy variables if any exist
        X = pd.get_dummies(X, drop_first=True)
        
        # Store the features used in session state
        st.session_state.features_used = X.columns.tolist()
        st.session_state.target_var = target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Store the trained model in session state
        st.session_state.trained_model = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display results
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{mse:.2f}")
        col2.metric("R-squared Score", f"{r2:.2f}")
        
        # Actual vs Predicted plot
        if len(y_test) > 1:
            fig7, ax7 = plt.subplots(figsize=(8, 4))
            ax7.scatter(y_test, y_pred)
            ax7.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            ax7.set_xlabel('Actual')
            ax7.set_ylabel('Predicted')
            ax7.set_title('Actual vs Predicted Values')
            st.pyplot(fig7)
        
        st.subheader("Coefficients")
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        })
        st.write(coef_df)
        
        # Feature importance plot
        if len(coef_df) > 0:
            fig8, ax8 = plt.subplots(figsize=(8, 4))
            ax8.barh(coef_df['Feature'], coef_df['Coefficient'])
            ax8.set_title('Feature Importance (Coefficient Values)')
            ax8.set_xlabel('Coefficient Value')
            st.pyplot(fig8)

with tab4:
    st.header("Prediction Results")
    
    if st.session_state.trained_model is not None:
        if submitted:
            # Convert gender to numerical value
            gender_encoded = 0 if gender == "Female" else 1
            
            # Create input DataFrame with same structure as training data
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender_encoded],
                'Weight': [weight],
                'Height': [height],
                'Fruit_Consumption': [fruit],
                'Vegetable_Consumption': [vegetable],
                'Water_Intake': [water],
                'Stress_Level': [stress]
            })
            
            # Use scaler fitted on training data
            scaler = MinMaxScaler()
            scaler.fit(df[['Fruit_Consumption', 'Vegetable_Consumption', 'Water_Intake']])  # Fit on full dataset

            nutrition_components = input_data[['Fruit_Consumption', 'Vegetable_Consumption', 'Water_Intake']]
            scaled_components = scaler.transform(nutrition_components)  # Transform only
            input_data['Nutrition_Score'] = scaled_components.mean(axis=1) * 10

            
            # Calculate BMI
            input_data['BMI'] = input_data['Weight'] / ((input_data['Height']/100) ** 2)
            
            # Ensure we only use the features that were used in training
            try:
                # Handle dummy variables if any were created during training
                if any('_' in feature for feature in st.session_state.features_used):
                    input_data = pd.get_dummies(input_data, drop_first=True)
                    for feature in st.session_state.features_used:
                        if feature not in input_data.columns:
                            input_data[feature] = 0
                    input_data = input_data[st.session_state.features_used]
                else:
                    input_data = input_data[st.session_state.features_used]
                
                # Make prediction
                prediction = st.session_state.trained_model.predict(input_data)
                st.success(f"Predicted {st.session_state.target_var}: {prediction[0]:.2f}")
                
                # Show nutrition metrics
                st.subheader("Your Nutrition Metrics")
                col1, col2 = st.columns(2)
                col1.metric("Nutrition Score", f"{input_data['Nutrition_Score'].values[0]:.1f}/10")
                col2.metric("BMI", f"{input_data['BMI'].values[0]:.1f}")
                
                # Show input data
                st.subheader("Your Input Data")
                st.write(input_data)
                        
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        else:
            st.info("Please submit the form to see predictions")
    else:
        st.warning("Please train a model in the 'Regression Model' tab firs")