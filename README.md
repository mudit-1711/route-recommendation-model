# AI Travel Route Recommendation System

This is an end-to-end Deep Learning-based travel route recommendation system. It intelligently combines 4 transportation datasets (Flights, Cabs, Trains, Buses) to predict and recommend the best travel routes between cities. It supports multi-modal combinations (e.g., Cab -> Flight -> Cab) and optimizes based on user preferences like "Cheapest", "Fastest", and "Comfort optimized".

## Project Structure

1. **`preprocess.py`**: Cleans data, imputes missing values, extracts date features, encodes categorical features, and scales numerical values. Artifacts (`scaler.pkl`, `encoders.pkl`) are saved in the `artifacts/` folder.
2. **`model.py`**: Defines the `HybridRecommender` PyTorch model. Uses categorical embeddings and continuous dense layers to output predictions for Price, Duration, and Comfort.
3. **`train.py`**: Loads preprocessed data and trains the model with Early Stopping. The final model is saved as `artifacts/model.pth`.
4. **`recommend.py`**: The core engine. It generates candidate routes (direct and multi-modal) for a given query and scores them using the trained model and user preferences.
5. **`test.py`**: A modern Streamlit application that provides an interactive UI for users to find their optimal routes.
6. **`requirements.txt`**: List of Python dependencies.

## Setup Instructions

1. Ensure you have Python 3.8+ installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing pipeline to generate data artifacts:
   ```bash
   python preprocess.py
   ```
4. Train the Deep Learning model:
   ```bash
   python train.py
   ```
5. Launch the Streamlit web application:
   ```bash
   streamlit run test.py
   ```

## Model Architecture
- **Tabular Embeddings**: Used for `Transport_Type`, `Mode`, `Source`, and `Destination`.
- **Continuous Features**: Scaled values for `Distance`, `Day`, `Month`, `Weekday`, `Is_Weekend`.
- **Multi-task Output**: The model simultaneously predicts the `Price`, `Duration`, and `Comfort` for any given route leg.

## Recommendation Logic
User preferences weigh the predicted metrics:
- **Cheapest**: Minimizes predicted price.
- **Fastest**: Minimizes predicted duration.
- **Economical**: A balanced minimization of price and duration.
- **Balanced**: Balances price, duration, and comfort.
- **Luxury**: Prioritizes premium transport and shorter duration.
- **Comfort optimized**: Strongly prioritizes the comfort metric.
