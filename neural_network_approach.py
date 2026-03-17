import pandas as pd
import pysr
import uproot
from pysr import PySRRegressor
import tensorflow as tf
import numpy as np
import warnings
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def predict_in_batches(model, x_data, y_data, batch_size=32):

    residuals = []
    
    for i in range(0, len(x_data), batch_size):
        x_batch = x_data[i:i+batch_size]
        y_batch_true = y_data[i:i+batch_size]
        y_batch_pred = model.predict(x_batch).flatten()

        batch_residuals = y_batch_true - y_batch_pred
        residuals.extend(batch_residuals)

    return np.array(residuals)

def modified_mse_loss(y_true, y_pred, weights): 
    mse_loss = tf.reduce_mean((y_true - y_pred) ** 2) 
    reg_loss = tf.reduce_sum([tf.reduce_sum(w ** 2) for w in weights]) 
    loss = mse_loss + 0.001 * reg_loss 
    return loss
    
def build_model(input_dim, learning_rate=0.001):

        model = Sequential()

        model.add(Dense(100, activation='relu', input_shape = (input_dim,)))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))

        opt = optimizers.Adam(learning_rate)
    
        model.compile(loss=lambda y_true, y_pred: modified_mse_loss(y_true, y_pred, model.trainable_variables), optimizer=opt, metrics=['mse'])

        return model


if __name__=='__main__':

    pthat_ranges = [
        (5.0, 7.0), (7.0, 9.0), (9.0, 12.0), (12.0, 16.0),
        (16.0, 21.0), (21.0, 28.0), (28.0, 36.0), (36.0, 45.0),
        (45.0, 57.0), (57.0, 70.0), (70.0, 85.0)
    ]

    symb_regr=False
    check=True

    tf.random.set_seed(20)
    warnings.filterwarnings("ignore")

    print("Reading the Training CSV File...")
    df = pd.read_csv("jetpt_training_sample.csv")
    threshold = 1e-10

    df_clean = df[(df.abs() >= threshold).all(axis=1)]
    df_sampled = df_clean.sample(n=400000, random_state=42)
    
    y = df_sampled.iloc[:, 0].values.astype(np.float32)
    x = df_sampled.iloc[:, 1:].values.astype(np.float32)

    x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.50, random_state=42)

    true_pt_bins = np.linspace(0, 100, 100)  # adjust the range if needed
    H_pt, edges_pt = np.histogram(y_train, bins=true_pt_bins)

    input_dim = x_train.shape[1]
    model=build_model(input_dim, learning_rate=0.0001)
    model.summary()

    print("Starting model training...")
    history = model.fit(x_train, y_train, epochs=100, batch_size=24, validation_split = 0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta = 0.001, restore_best_weights=True)])

    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.show()

    loss, mse = model.evaluate(x_test, y_test)
    print(f"\nTest MSE: {mse:.4f}")

    residuals_array = predict_in_batches(model, x_test, y_test, batch_size=32)

    plt.hist(residuals_array, bins=50, alpha=0.7, label='Histogram')
    plt.xlim(-50, 50)
    plt.show()

    with uproot.recreate("residuals_modified_mse.root") as file:

        file["truePt_training"] = (H_pt, edges_pt)

        if check:

            for i, (low, high) in enumerate(pthat_ranges, start=1):

                filename = f"jetpt_testing_sample_pthat_{low}_{high}.csv"
                df = pd.read_csv(filename)

                df_clean = df[(df.abs() >= threshold).all(axis=1)]
                y_true = df_clean.iloc[:, 0].values.astype(np.float32)
                x_data = df_clean.iloc[:, 1:].values.astype(np.float32)
                y_pred = model.predict(x_data).flatten()
                residuals = y_true - y_pred

                residual_bins = np.linspace(-50, 50, 100)   # 100 bins for residuals
                true_pt_bins  = np.linspace(0,  100, 100)    # adjust upper bound to your true jet pt range

                H, xedges, yedges = np.histogram2d(y_true, residuals,bins=[true_pt_bins, residual_bins])

                hist_name = f"residuals_vs_truePt_pthat_{low}_{high}"
                file[hist_name] = (H, xedges, yedges)

                print(f"Processed pthat {low}-{high}, mean={residuals.mean():.4f}, std={residuals.std():.4f}")

    print("\nAll 2D residual histograms written to residuals_modified_mse.root")


    if symb_regr:
        y_pred_nn = model.predict(x_train).flatten()

        symbolic_model = PySRRegressor( niterations=50, populations=20, population_size=33, binary_operators=["+", "-", "*", "/", "^"], unary_operators=["exp", "log", "sin", "cos", "tan", "sqrt"], model_selection="best", maxsize=20, verbosity=1)

        symbolic_model.fit(x_train, y_pred_nn)

        print(symbolic_model)

        y_symbolic = symbolic_model.predict(x_test)
        residuals = y_test - y_symbolic

        plt.hist(residuals, bins=50, alpha=0.7, label='Histogram')
        plt.xlim(-50, 50)
        plt.show()

    

