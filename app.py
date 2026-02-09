"""
Simple AI/ML Bank Recruitment Prediction App
Workshop Demo - Streamlit + Scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. SYNTHETIC TRAINING DATA (20 rows)
# ============================================

def create_training_data():
    """
    Create synthetic training data with Indonesian universities
    """
    data = {
        'name': [
            'Budi Santoso', 'Siti Nurhaliza', 'Ahmad Dhani', 'Rina Wijaya', 'Dimas Prakoso',
            'Ayu Lestari', 'Rudi Hermawan', 'Maya Putri', 'Eko Saputra', 'Dewi Anggraini',
            'Fajar Ramadhan', 'Linda Susanti', 'Arief Wibowo', 'Nadia Kartika', 'Hendra Gunawan',
            'Putri Amelia', 'Yoga Pratama', 'Sari Indah', 'Bambang Setiawan', 'Lia Permata'
        ],
        'university': [
            'Universitas Indonesia', 'Institut Teknologi Bandung', 'Universitas Gadjah Mada',
            'Universitas Airlangga', 'Institut Teknologi Bandung', 'Universitas Indonesia',
            'Universitas Brawijaya', 'Universitas Gadjah Mada', 'Universitas Diponegoro',
            'Universitas Indonesia', 'Universitas Padjadjaran', 'Institut Teknologi Bandung',
            'Universitas Hasanuddin', 'Universitas Indonesia', 'Universitas Sebelas Maret',
            'Universitas Gadjah Mada', 'Universitas Bina Nusantara', 'Universitas Airlangga',
            'Universitas Pelita Harapan', 'Institut Teknologi Bandung'
        ],
        'gpa': [3.75, 3.82, 3.45, 3.20, 3.68, 3.91, 3.15, 3.55, 3.30, 3.78,
                3.42, 3.88, 3.25, 3.95, 3.18, 3.72, 3.65, 3.40, 3.10, 3.85],
        'faculty': [
            'Ekonomi', 'Teknik Industri', 'Manajemen', 'Akuntansi', 'Teknik Informatika',
            'Ekonomi', 'Pertanian', 'Manajemen', 'Hukum', 'Ekonomi',
            'Ilmu Komunikasi', 'Teknik Industri', 'Kedokteran', 'Ekonomi', 'Sastra',
            'Manajemen', 'Sistem Informasi', 'Akuntansi', 'Teknik Sipil', 'Matematika'
        ],
        'city': [
            'Jakarta', 'Bandung', 'Yogyakarta', 'Surabaya', 'Bandung',
            'Jakarta', 'Malang', 'Yogyakarta', 'Semarang', 'Jakarta',
            'Bandung', 'Bandung', 'Makassar', 'Jakarta', 'Surakarta',
            'Yogyakarta', 'Jakarta', 'Surabaya', 'Tangerang', 'Bandung'
        ],
        'accepted': [
            1, 1, 0, 0, 1, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 1, 0, 0, 1
        ]  # 1 = Accepted, 0 = Rejected
    }
    
    return pd.DataFrame(data)

# ============================================
# 2. FEATURE ENGINEERING
# ============================================

def engineer_features(df):
    """
    Create derived features for better prediction
    """
    # Top-tier universities (typically better acceptance rate)
    top_universities = [
        'Universitas Indonesia', 
        'Institut Teknologi Bandung', 
        'Universitas Gadjah Mada'
    ]
    df['top_university'] = df['university'].isin(top_universities).astype(int)
    
    # Business-related faculties (relevant for banking)
    business_faculties = ['Ekonomi', 'Manajemen', 'Akuntansi']
    df['business_faculty'] = df['faculty'].isin(business_faculties).astype(int)
    
    # Major financial hubs
    major_cities = ['Jakarta', 'Surabaya', 'Bandung']
    df['major_city'] = df['city'].isin(major_cities).astype(int)
    
    # GPA categories
    df['high_gpa'] = (df['gpa'] >= 3.5).astype(int)
    
    return df

# ============================================
# 3. MODEL TRAINING
# ============================================

@st.cache_resource
def train_model():
    """
    Train the recruitment prediction model
    Cache it so it doesn't retrain on every interaction
    """
    # Load training data
    df = create_training_data()
    
    # Engineer features
    df = engineer_features(df)
    
    # Select features for training
    feature_columns = ['gpa', 'top_university', 'business_faculty', 'major_city', 'high_gpa']
    X = df[feature_columns]
    y = df['accepted']
    
    # Train model (using all data since we only have 20 rows)
    # For demo purposes, we'll use Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    # Calculate training accuracy
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    return model, df, feature_columns, accuracy

# ============================================
# 4. PREDICTION FUNCTION
# ============================================

def make_prediction(model, university, gpa, faculty, city):
    """
    Make prediction for a new candidate
    """
    # Create feature dictionary
    top_universities = [
        'Universitas Indonesia', 
        'Institut Teknologi Bandung', 
        'Universitas Gadjah Mada'
    ]
    business_faculties = ['Ekonomi', 'Manajemen', 'Akuntansi']
    major_cities = ['Jakarta', 'Surabaya', 'Bandung']
    
    features = {
        'gpa': gpa,
        'top_university': 1 if university in top_universities else 0,
        'business_faculty': 1 if faculty in business_faculties else 0,
        'major_city': 1 if city in major_cities else 0,
        'high_gpa': 1 if gpa >= 3.5 else 0
    }
    
    # Create dataframe for prediction
    X_new = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0]
    
    return prediction, probability

# ============================================
# 5. STREAMLIT UI
# ============================================

def main():
    # Page config
    st.set_page_config(
        page_title="Bank Recruitment Predictor",
        page_icon="🏦",
        layout="wide"
    )
    
    # Title
    st.title("🏦 Bank Recruitment Prediction System")
    st.markdown("### AI/ML Workshop Demo - Predicting Candidate Acceptance")
    st.markdown("---")
    
    # Train model
    with st.spinner("Training AI model..."):
        model, training_data, feature_columns, accuracy = train_model()
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Candidate Information")
        
        # Input form
        with st.form("candidate_form"):
            name = st.text_input("Full Name", placeholder="e.g., Budi Santoso")
            
            university = st.selectbox(
                "University",
                options=[
                    'Universitas Indonesia',
                    'Institut Teknologi Bandung',
                    'Universitas Gadjah Mada',
                    'Universitas Airlangga',
                    'Universitas Brawijaya',
                    'Universitas Diponegoro',
                    'Universitas Padjadjaran',
                    'Universitas Hasanuddin',
                    'Universitas Sebelas Maret',
                    'Universitas Bina Nusantara',
                    'Universitas Pelita Harapan',
                    'Other'
                ]
            )
            
            gpa = st.number_input(
                "GPA (0.00 - 4.00)",
                min_value=0.0,
                max_value=4.0,
                value=3.5,
                step=0.01,
                format="%.2f"
            )
            
            faculty = st.selectbox(
                "Faculty/Major",
                options=[
                    'Ekonomi',
                    'Manajemen',
                    'Akuntansi',
                    'Teknik Industri',
                    'Teknik Informatika',
                    'Sistem Informasi',
                    'Matematika',
                    'Ilmu Komunikasi',
                    'Hukum',
                    'Teknik Sipil',
                    'Other'
                ]
            )
            
            city = st.selectbox(
                "City",
                options=[
                    'Jakarta',
                    'Surabaya',
                    'Bandung',
                    'Yogyakarta',
                    'Semarang',
                    'Malang',
                    'Makassar',
                    'Surakarta',
                    'Tangerang',
                    'Other'
                ]
            )
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Acceptance", use_container_width=True)
        
        # Make prediction when form is submitted
        if submitted:
            if name:
                prediction, probability = make_prediction(model, university, gpa, faculty, city)
                
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                # Display result with nice formatting
                if prediction == 1:
                    st.success(f"✅ **{name}** is predicted to be **ACCEPTED**")
                    st.metric(
                        "Acceptance Probability",
                        f"{probability[1]*100:.1f}%",
                        delta="High Chance"
                    )
                else:
                    st.error(f"❌ **{name}** is predicted to be **REJECTED**")
                    st.metric(
                        "Acceptance Probability",
                        f"{probability[1]*100:.1f}%",
                        delta="Low Chance",
                        delta_color="inverse"
                    )
                
                # Show probability breakdown
                st.markdown("### Probability Breakdown")
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.info(f"🚫 Rejection: **{probability[0]*100:.1f}%**")
                with prob_col2:
                    st.success(f"✅ Acceptance: **{probability[1]*100:.1f}%**")
                
            else:
                st.warning("⚠️ Please enter the candidate's name.")
    
    with col2:
        st.subheader("📊 Model Information")
        
        # Model accuracy
        st.metric("Training Accuracy", f"{accuracy*100:.1f}%")
        
        st.info(f"""
        **Model Type:** Logistic Regression
        
        **Training Data:** {len(training_data)} candidates
        
        **Features Used:**
        - GPA Score
        - University Tier
        - Faculty Relevance
        - City Location
        - GPA Category
        """)
        
        # Show training data statistics
        with st.expander("📈 Training Data Statistics"):
            st.write(f"**Accepted:** {training_data['accepted'].sum()} candidates")
            st.write(f"**Rejected:** {len(training_data) - training_data['accepted'].sum()} candidates")
            st.write(f"**Average GPA:** {training_data['gpa'].mean():.2f}")
            st.write(f"**Top Universities:** {training_data['top_university'].sum()} candidates")
        
        # Disclaimer
        st.warning("""
        ⚠️ **Disclaimer**
        
        This is a demo model for educational purposes only. 
        
        With only 20 training samples, predictions should not be used for real recruitment decisions.
        """)
    
    # Show raw training data at the bottom
    with st.expander("🔍 View Training Data"):
        st.dataframe(training_data, use_container_width=True)

if __name__ == "__main__":
    main()
