# HK-16_SUPERNOVA_AGROPREDICT
AgroPredict – Geospatial AI-Based Agricultural Decision Support System

AgroPredict is a real-time, geospatial AI-powered agricultural decision-support system that integrates environmental data with machine learning to provide crop recommendations and post-harvest spoilage risk prediction.
The platform combines live weather, soil, and climate data with a fault-tolerant ML pipeline to enable data-driven farming decisions.

🚀 Project Overview
Agricultural decisions are often made using fragmented data or manual estimation. AgroPredict bridges this gap by:
Extracting real-time environmental parameters using APIs
Processing and cleaning raw geospatial data
Converting it into structured feature vectors
Feeding the data into trained ML models
Generating spoilage risk and crop insights

The system supports both automatic data fetching and manual input override, ensuring flexibility and reliability.
🌍 Key Features
🌐 Interactive global map-based location selection
🌦 OpenWeather API integration for live weather data
🌱 SoilGrids API integration for soil properties
☀ NASA POWER API integration for climate data
🔄 Real-time multi-source API fusion
🧹 Data cleaning and sentinel value (-999) handling
🔁 Longitude normalization for global map wrap correction
🧠 Structured ML feature vector generation
⚙ Fault-tolerant partial auto-fill system
✍ Manual override capability
📊 Streamlit-based interactive dashboard

🧠 How It Works
User selects a location via interactive map.
System fetches real-time weather, soil, and climate data.
Raw API data is cleaned and normalized.
Structured feature vectors are created.
ML model predicts:
Spoilage risk
Crop suitability
Results are displayed on an interactive dashboard.

🏗 Architecture Overview
User Input (Map / Manual)
        ↓
Geospatial API Layer
(Weather + Soil + Climate)
        ↓
Data Cleaning & Normalization
        ↓
Feature Engineering
        ↓
Machine Learning Model
        ↓
Prediction & Dashboard Output

🎯 Innovation & Uniqueness
Hybrid auto-fetch + manual override architecture
Robust fault-tolerant design
Real-time geospatial data integration
Transparent ML workflow (not black-box)
Lightweight and scalable Streamlit deployment

💡 Use Cases

Farmers seeking data-driven crop planning
FPOs (Farmer Producer Organizations)
Agricultural markets and mandis
Academic research in agri-tech
Government agricultural advisory systems

🛠 Tech Stack
Python
Streamlit
Pandas / NumPy
Scikit-learn (ML Models)
OpenWeather API
SoilGrids API
NASA POWER API
Folium (Map Integration)

📈 Future Enhancements
Market price forecasting integration
Crop rotation optimization
Yield prediction module
Mobile application deployment

Government agricultural network integrationAgroPredict is a geospatial AI-powered agricultural decision-support system that integrates real-time weather, soil, and climate data to generate crop recommendations and spoilage risk predictions through a fault-tolerant ML pipeline.
