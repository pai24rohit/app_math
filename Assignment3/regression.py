import torch
from torch import nn
# Data: Number of Gold, Number of Silver, Value (GBP)
data = [
    [24.0, 2.0, 1422.40],
    [24.0, 4.0, 1469.50],
    [16.0, 3.0, 1012.70],
    [25.0, 6.0, 1632.20],
    [16.0, 1.0, 952.20],
    [19.0, 2.0, 1117.70],
    [14.0, 3.0, 906.20],
    [22.0, 2.0, 1307.30],
    [25.0, 4.0, 1552.80],
    [12.0, 1.0, 686.70],
    [24.0, 7.0, 1543.40],
    [19.0, 1.0, 1086.50],
    [23.0, 7.0, 1495.20],
    [19.0, 5.0, 1260.70],
    [21.0, 3.0, 1288.10],
    [16.0, 6.0, 1111.50],
    [24.0, 5.0, 1523.10],
    [19.0, 7.0, 1297.40],
    [14.0, 4.0, 946.40],
    [20.0, 3.0, 1197.10]
]

# Convert data to tensors
X = torch.tensor([[item[0], item[1]] for item in data], dtype=torch.float32)
y = torch.tensor([item[2] for item in data], dtype=torch.float32).unsqueeze(1)


def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    Hint: use nn.Linear
    """
    model = nn.Linear(input_size, output_size)
    return model


def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    Hint: use the train_iteration function.
    Hint 2: while woring you can use the print function to print the loss every 1000 epochs.
    Hint 3: you can use the previos_loss variable to stop the training when the loss is not changing much.
    """
    learning_rate = 0.001 # Lower learning rate
    num_epochs = 5000 # Pick a better number of epochs
    input_features = X.shape[1] # extract the number of features from the input `shape` of X
    output_features = y.shape[1] # extract the number of features from the output `shape` of y
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss() # Mean squared error loss

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previos_loss = float("inf")

    prev_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        if abs(prev_loss - loss.item()) < 1e-6:  # Stopping condition
            break
        prev_loss = loss.item()

    return model, loss
# Example predictions to test the model
if __name__ == "__main__":
    def get_train_data(dim=1):
        """
        dim is the number of features in the input. for our purposes it will be either 1 or 2.
        """
        X_2 = torch.tensor(
            [[24.,  2.],
             [24.,  4.],
             [16.,  3.],
             [25.,  6.],
             [16.,  1.],
             [19.,  2.],
             [14.,  3.],
             [22.,  2.],
             [25.,  4.],
             [12.,  1.],
             [24.,  7.],
             [19.,  1.],
             [23.,  7.],
             [19.,  5.],
             [21.,  3.],
             [16.,  6.],
             [24.,  5.],
             [19.,  7.],
             [14.,  4.],
             [20.,  3.]])
        y = torch.tensor(
            [[1422.4000],
             [1469.5000],
             [1012.7000],
             [1632.2000],
             [952.2000],
             [1117.7000],
             [906.2000],
             [1307.3000],
             [1552.8000],
             [686.7000],
             [1543.4000],
             [1086.5000],
             [1495.2000],
             [1260.7000],
             [1288.1000],
             [1111.5000],
             [1523.1000],
             [1297.4000],
             [946.4000],
             [1197.1000]])
        if dim == 1:
            X = X_2[:, :1]
        elif dim == 2:
            X = X_2
        else:
            raise ValueError("dim must be 1 or 2")
        return X, y

    X, y = get_train_data(dim=2)
    model, loss = fit_regression_model(X, y)
    print(f"Final Loss: {loss.item()}")

    def predict_value(model, gold, silver):
        with torch.no_grad():
            input_tensor = torch.tensor([[gold, silver]], dtype=torch.float32)
            predicted_value = model(input_tensor).item()
        return predicted_value

    # Example predictions
    print(f"Predicted value for a vault with 30 gold and 5 silver: {predict_value(model, 30, 5)} GBP")
    print(f"Predicted value for a vault with 10 gold and 2 silver: {predict_value(model, 10, 2)} GBP")

