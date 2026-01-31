import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sales.models import Transaction
from datetime import timedelta

class SalesForecaster:
    """
    Predictive Analytics Engine.
    """
    
    @staticmethod
    def predict_next_30_days():
        """
        Uses Linear Regression to forecast future sales.
        Prophet would be better but requires complex install (C++ compilers).
        Sklearn is safer for this environment.
        """
        # 1. Get Historical Data
        queryset = Transaction.objects.filter(status='final').values('transaction_date', 'final_total')
        if not queryset.exists():
            return []

        df = pd.DataFrame(list(queryset))
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Aggregate by day
        daily = df.groupby(df['transaction_date'].dt.date)['final_total'].sum().reset_index()
        daily.columns = ['date', 'revenue']
        
        if len(daily) < 2:
            return [] # Not enough data
            
        # 2. Prepare Features (X = Day Number, y = Revenue)
        daily['day_num'] = (pd.to_datetime(daily['date']) - pd.to_datetime(daily['date']).min()).dt.days
        X = daily[['day_num']]
        y = daily['revenue']
        
        # 3. Train Model
        model = LinearRegression()
        model.fit(X, y)
        
        # 4. Predict Future
        last_day = daily['day_num'].max()
        future_days = np.array(range(last_day + 1, last_day + 31)).reshape(-1, 1)
        predictions = model.predict(future_days)
        
        results = []
        base_date = pd.to_datetime(daily['date']).min()
        
        for i, pred in enumerate(predictions):
            date = base_date + timedelta(days=int(future_days[i][0]))
            results.append({
                "date": date.strftime('%Y-%m-%d'),
                "predicted_revenue": max(0, float(pred)) # No negative revenue
            })
            
        return results
