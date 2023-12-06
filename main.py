import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# ダミーデータ生成
def generate_images(input_params):
    # XYZ値の生成則に基づき、各ピクセルごとにダミー画像を生成
    images = []
    for params in input_params:
        image = np.zeros((10, 10, 3))  # 10x10ピクセルのRGB画像
        temperature, humidity, speed = params

        # 仮の生成則：各ピクセルのRGB値を入力パラメータから計算（実際の生成則に置き換えてください）
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                r = temperature + np.random.normal(0, 0.01)
                g = humidity + np.random.normal(0, 0.01)
                b = speed + np.random.normal(0, 0.01)
                image[i, j] = [r, g, b]

        images.append(image)
    return np.array(images)

# モデルの定義
def create_model():
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(3,)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10 * 10 * 3, activation='sigmoid'))  # Flatten the output
    model.add(layers.Reshape((10, 10, 3)))  # Reshape to match image dimensions
    return model

# ロス関数の定義
def custom_rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.keras.backend.flatten(y_true) - tf.keras.backend.flatten(y_pred))))

# ロスの可視化
def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# 画像比較と可視化
def compare_images(model, input_params, true_images):
    predicted_images = model.predict(input_params)
    for i in range(len(input_params)):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # 本来の画像
        axes[0].imshow(true_images[i])
        axes[0].set_title('True Image')

        # モデルによる生成画像
        axes[1].imshow(predicted_images[i])
        axes[1].set_title('Generated Image')

        plt.show()

# XY値の散布図を比較
def compare_scatter(input_params, true_images, model):
    predicted_images = model.predict(input_params)

    for i in range(len(input_params)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 本来の画像
        axes[0].imshow(true_images[i])
        axes[0].set_title('True Image')

        # モデルによる生成画像
        axes[1].imshow(predicted_images[i])
        axes[1].set_title('Generated Image')

        # XY値の散布図
        axes[2].scatter(true_images[i].reshape(-1, 3)[:, 0], true_images[i].reshape(-1, 3)[:, 1], label='True Data', alpha=0.5)
        axes[2].scatter(predicted_images[i].reshape(-1, 3)[:, 0], predicted_images[i].reshape(-1, 3)[:, 1], label='Predicted Data', alpha=0.5)
        axes[2].set_title('XY Scatter Plot')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        axes[2].legend()

        plt.show()

# メインプロセス
if __name__ == "__main__":
    # モデルの作成
    model = create_model()

    # ダミーデータの生成
    num_samples = 1000
    input_params = np.random.rand(num_samples, 3)
    true_images = generate_images(input_params)

    # モデルの訓練
    model.compile(optimizer='adam', loss=custom_rmse_loss)
    history = model.fit(input_params, true_images, epochs=50, validation_split=0.2, verbose=1)

    # ロスの可視化
    plot_loss(history)

    # 学習結果の検証
    compare_images(model, input_params[:5], true_images[:5])
    
    # XY値の散布図の比較
    compare_scatter(input_params, true_images, model)
