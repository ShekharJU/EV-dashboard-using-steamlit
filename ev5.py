import streamlit as st
import pandas as pd
import plotly.express as px  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
st.set_page_config(layout="wide")
st.title('🚗 Electric Vehicle Population Dashboard')

file = st.file_uploader("Upload CSV file", type=['csv'])
if file:
    df = pd.read_csv(file)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Fill missing values
df.fillna({'Electric Range': 0, 'Make': 'Unknown', 'Model': 'Unknown'}, inplace=True)

# Sidebar Filters
with st.sidebar:
    st.header('🔍 Filter Data')

    # Select Model Year
    year_range = (int(df['Model Year'].min()), int(df['Model Year'].max()))
    selected_year = st.slider('Select Model Year', year_range[0], year_range[1], year_range[1])

    # Select Make
    makes = ['All'] + sorted(df['Make'].dropna().unique().tolist())
    selected_make = st.selectbox('Select Make', makes)

    # Filtered Data
    filtered_df = df[df['Model Year'] == selected_year]
    if selected_make != 'All':
        filtered_df = filtered_df[filtered_df['Make'] == selected_make]

    # Display Key Metrics
    st.metric("📌 Total Vehicles", len(filtered_df))
    st.metric("🔋 Avg. Electric Range", f"{filtered_df['Electric Range'].mean():.2f} miles")
    st.metric("🏆 Top Manufacturer", filtered_df['Make'].mode()[0] if not filtered_df.empty else "N/A")

# Display filtered data
with st.expander('📂 Filtered Data'):
    st.write(filtered_df)

# Data Visualizations
with st.expander('📈 Data Visualization'):
    st.subheader("Electric Range by Model Year")
    st.scatter_chart(data=filtered_df, x='Model Year', y='Electric Range', color='Make')

    # **Top 10 EV Makes**
    st.subheader("🔝 Top 10 EV Makes by Total Vehicles")
    top_makes = df['Make'].value_counts().nlargest(10).reset_index()
    top_makes.columns = ['Make', 'Total Vehicles']
    fig_make = px.bar(top_makes, x='Make', y='Total Vehicles', title="Top 10 EV Makes", text_auto=True)
    st.plotly_chart(fig_make)

    # **Top 10 EV Models**
    st.subheader("🚘 Top 10 EV Models by Total Vehicles")
    top_models = df['Model'].value_counts().nlargest(10).reset_index()
    top_models.columns = ['Model', 'Total Vehicles']
    fig_model = px.bar(top_models, x='Model', y='Total Vehicles', title="Top 10 EV Models", text_auto=True)
    st.plotly_chart(fig_model)

    # **Pie Chart for EV Type**
    st.subheader("🔄 Electric Vehicle Type Distribution")
    ev_type_counts = df['Electric Vehicle Type'].value_counts().reset_index()
    ev_type_counts.columns = ['Electric Vehicle Type', 'Count']
    fig_pie = px.pie(ev_type_counts, names='Electric Vehicle Type', values='Count', title="EV Type Distribution", hole=0.4)
    st.plotly_chart(fig_pie)

# Machine Learning Model
st.header('🧠 EV Classification Model')

# Data Preparation
encode_cols = ['County', 'City', 'State', 'Make', 'Model', 'Electric Vehicle Type', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility', 'Electric Utility']
existing_encode_cols = [col for col in encode_cols if col in df.columns]
df_encoded = pd.get_dummies(df, columns=existing_encode_cols, drop_first=True)

drop_columns = ['VIN (1-10)', 'DOL Vehicle ID', 'Vehicle Location', '2020 Census Tract']
drop_columns = [col for col in drop_columns if col in df_encoded.columns]
X = df_encoded.drop(columns=drop_columns)
y = df['Make']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.subheader("📊 Model Accuracy")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Predict based on a sample input
sample_input = X.iloc[:1]
prediction = clf.predict(sample_input)

st.subheader('🎯 Predicted Make')
st.success(prediction[0])
