import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime

class ForecastEngine:
    def __init__(self):
        self.model = LinearRegression()

    def predict_sales_next_7_days(self, daily_sales_data: list) -> dict:
        """
        Predicts total sales for the next 7 days based on history.
        input: list of {'date': 'YYYY-MM-DD', 'total': float}
        """
        if not daily_sales_data or len(daily_sales_data) < 3:
            return {"error": "Not enough data for prediction (need > 3 days)"}

        try:
            df = pd.DataFrame(daily_sales_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Convert dates to ordinal for regression
            df['date_ordinal'] = df['date'].map(datetime.datetime.toordinal)
            
            X = df[['date_ordinal']]
            y = df['total']
            
            # Fit Model
            self.model.fit(X, y)
            
            # Predict next 7 days
            last_date = df['date'].max()
            future_dates = [last_date + datetime.timedelta(days=x) for x in range(1, 8)]
            future_ordinals = [[d.toordinal()] for d in future_dates]
            
            predictions = self.model.predict(future_ordinals)
            # Ensure no negative sales
            predictions = [max(0, p) for p in predictions]
            
            total_predicted = sum(predictions)
            
            return {
                "total_predicted_7d": total_predicted,
                "daily_forecast": dict(zip([d.strftime('%Y-%m-%d') for d in future_dates], predictions)),
                "trend": "increasing" if self.model.coef_[0] > 0 else "decreasing"
            }
        except Exception as e:
            return {"error": str(e)}

    def predict_stockout_date(self, current_stock: float, sales_history: list) -> dict:
        """
        Predicts when stock will reach 0.
        input: current_stock (float), sales_history (list of daily qty sold)
        """
        if not sales_history:
             return {"days_left": 999, "date": "Unknown (No Data)"}
             
        avg_daily_sales = np.mean(sales_history)
        if avg_daily_sales <= 0:
            return {"days_left": 999, "date": "Never (No Sales)"}
            
        days_left = current_stock / avg_daily_sales
        
        # Cap at reasonable future
        if days_left > 365:
             return {"days_left": 365, "date": "> 1 Year"}
             
        stockout_date = datetime.date.today() + datetime.timedelta(days=int(days_left))
        
        return {
            "days_left": int(days_left),
            "date": stockout_date.strftime('%Y-%m-%d'),
            "burn_rate": float(avg_daily_sales)
        }
