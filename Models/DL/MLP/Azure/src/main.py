%%writefile {train_src_dir}/main.py
import os
import random
import numpy as np
import warnings
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow
import tensorflow as tf
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.callbacks import Callback
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Set seed for reproducibility
def init_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto()
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1)
    os.environ['TF_CUDNN_DETERMINISTIC'] = 'True'
    os.environ['TF_DETERMINISTIC_OPS'] = 'True'
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),
                                config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    return sess

init_seeds(seed=42)

def main():
    """Main function of the script."""

    # Input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, help='path to input train data')
    parser.add_argument('--test_data', type=str, help='path to input test data')
    parser.add_argument('--epochs', required=False, default=30, type=int)
    parser.add_argument('--batch_size', required=False, default=1, type=int)
    args = parser.parse_args()
   
    # Start Logging
    mlflow.start_run()

    ###################
    #<prepare the data>
    ###################
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    print('input train data:', args.train_data)
    print('input test data:', args.test_data)
    
    trainDF = pd.read_csv(args.train_data, low_memory=False)
    testDF = pd.read_csv(args.test_data, low_memory=False)

    train_label = trainDF[['price']]
    test_label = testDF[['price']]

    train_features = trainDF.drop(columns = ['price'])
    test_features = testDF.drop(columns = ['price'])

    train_features = pd.get_dummies(train_features, drop_first=True)
    test_features = pd.get_dummies(test_features, drop_first=True)

    sc = StandardScaler()
    train_features = pd.DataFrame(sc.fit_transform(train_features))
    test_features = pd.DataFrame(sc.transform(test_features))

    mlflow.log_metric('num_samples', train_features.shape[0])
    mlflow.log_metric('num_features', train_features.shape[1])

    print(f"Training with data of shape {train_features.shape}")

    ####################
    #</prepare the data>
    ####################

    ##################
    #<train the model>
    ##################
    model = Sequential()
    model.add(Dense(130, input_dim=53, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(130, kernel_initializer='normal', activation='relu'))
    model.add(Dense(130, kernel_initializer='normal', activation='relu'))
    model.add(Dense(70, kernel_initializer='normal', activation='relu'))
    model.add(Dense(130, kernel_initializer='normal', activation='relu'))
    model.add(Dense(70, kernel_initializer='normal', activation='relu'))
    model.add(Dense(130, kernel_initializer='normal', activation='relu'))
    model.add(Dense(70, kernel_initializer='normal', activation='relu'))
    model.add(Dense(130, kernel_initializer='normal', activation='relu'))
    model.add(Dense(70, kernel_initializer='normal', activation='relu'))
    model.add(Dense(130, kernel_initializer='normal', activation='relu'))
    model.add(Dense(130, kernel_initializer='normal', activation='relu'))
    model.add(Dense(130, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mae', metrics=['mse'], optimizer=opt)
    model.summary()

    # Log metrics
    class LogRunMetrics(Callback):
        # Callback at the end of every epoch
        def on_epoch_end(self, epoch, log):
            # log a value repeated which creates a list
            mlflow.log_metric('Loss', log['loss'])
            mlflow.log_metric('Val_Loss', log['val_loss'])

    log_folder = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    filepath = 'MLP_weights_only_HPO1_bestModel.tf'
    checkpoint_dir = os.path.dirname(filepath)

    callbacks_list = [EarlyStopping(monitor='val_loss', patience=5), 
                      ModelCheckpoint(filepath, monitor='mse',
                                      save_best_only=True, mode='min'),
                      LogRunMetrics()]

    history = model.fit(train_features, train_label, epochs=args.epochs, 
                        batch_size=args.batch_size, validation_split=0.2, 
                        callbacks=callbacks_list)

    model.save('./MLP_HPO1_bestModel', save_format='tf')

    # Load model for more training or later use
    #filepath = 'MLP_weights_only_b4_HPO1_bestModel.h5'
    #model = tf.keras.models.load_model('./MLP_HPO1_bestModel_tf.h5')
    #model.load_weights(filepath)

    ##################
    #</train the model>
    ##################

    #####################
    #<evaluate the model>
    ##################### 
    plt.title('Model Error for Price')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylabel('Error [Price]')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('Loss vs. Price.png')
    mlflow.log_artifact('Loss vs. Price.png')
    plt.close()

    losses = pd.DataFrame(model.history.history)
    losses.plot()
    plt.title('Model Error for Price')
    plt.ylabel('Error [Price]')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('Error vs. Price.png')
    mlflow.log_artifact('Error vs. Price.png')
    plt.close()

    pred_train = model.predict(train_features)

    # Metrics: Train set
    train_mae = mean_absolute_error(train_label[:], pred_train[:])
    train_mse = mean_squared_error(train_label[:], pred_train[:])
    train_rmse = np.sqrt(mean_squared_error(train_label[:], pred_train[:]))
    train_r2 = r2_score(train_label[:], pred_train[:])

    pred_test = model.predict(test_features)

    # Metrics: Test set
    test_mae = mean_absolute_error(test_label[:], pred_test[:])
    test_mse = mean_squared_error(test_label[:], pred_test[:])
    test_rmse = np.sqrt(mean_squared_error(test_label[:], pred_test[:]))
    test_r2 = r2_score(test_label[:], pred_test[:])

    mlflow.log_metric('train_mae', train_mae)
    mlflow.log_metric('train_mse', train_mse)
    mlflow.log_metric('train_rmse', train_rmse)
    mlflow.log_metric('train_r2', train_r2)
    mlflow.log_metric('test_mae', test_mae)
    mlflow.log_metric('test_mse', test_mse)
    mlflow.log_metric('test_rmse', test_rmse)
    mlflow.log_metric('test_r2', test_r2)

    MaximumPrice = np.amax(test_label)
    PredictedMaxPrice = np.amax(pred_test)
    AveragePrice = np.average(test_label)
    PredictedAveragePrice = np.average(pred_test)
    MinimumPrice = np.amin(test_label)
    PredictedMinimumPrice = np.amin(pred_test)

    mlflow.log_metric('Maximum Price', MaximumPrice)
    mlflow.log_metric('Predicted Maximum Price', PredictedMaxPrice)
    mlflow.log_metric('Average Price', AveragePrice)
    mlflow.log_metric('Predicted Average Price', PredictedAveragePrice)
    mlflow.log_metric('Minimum Price', MinimumPrice)
    mlflow.log_metric('Predicted Minimum Price', PredictedMinimumPrice)

    ###################
    #</evaluate the model>
    ###################
    
    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()