import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Load and preprocess data
df = pd.read_csv('sunspots.csv', parse_dates=['Date'], index_col='Date')
df.rename(columns={'Monthly Mean Total Sunspot Number': 'Sunspots'}, inplace=True)

# Scale the data
scaler = MinMaxScaler()
sunspots_scaled = scaler.fit_transform(df[['Sunspots']].values)

# 2. Create sequences
SEQ_LENGTH = 20

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(sunspots_scaled, SEQ_LENGTH)

# 3. Train/test split (80% train)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# 4. Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))  # LSTM output
        out = out[:, -1, :]  # Take last timestep
        out = self.linear(out)  # Linear layer output
        return out

# 5. Initialize model, loss, optimizer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6. Train the model
EPOCHS = 50
for epoch in range(EPOCHS):
    start_time = time.time()
    
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    end_time = time.time()
    epoch_time = end_time - start_time
    
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}, Time: {epoch_time:.2f} sec")

# 7. Evaluate on test data
model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()

# Inverse transform to original scale
predicted_sunspots = scaler.inverse_transform(predictions)
actual_sunspots = scaler.inverse_transform(y_test.numpy())

# 8. Plot results
plt.figure(figsize=(12,6))
plt.plot(actual_sunspots, label='Actual')
plt.plot(predicted_sunspots, label='LSTM Forecast')
plt.title('Sunspot Forecast - LSTM')
plt.xlabel('Time Step')
plt.ylabel('Monthly Mean Sunspot Number')
plt.legend()
plt.grid()
plt.show()
