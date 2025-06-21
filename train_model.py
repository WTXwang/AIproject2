import numpy as np
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import os
# 定义模型保存路径
# 如果在同一目录下运行，确保train_model.py和mnist_cnn_model.keras在同一目录下
# 否则需要修改路径
MODEL_PATH = 'mnist_cnn_model.keras'

def train_and_save_model():
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # 数据预处理：将图像数据归一化到0-1范围，并调整形状以适应卷积层输入
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    # 将数据形状调整为 (样本数, 高度, 宽度, 通道数)
    # MNIST数据集是灰度图像，所以通道数为1
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # 构建卷积神经网络模型
    model = keras.Sequential([
        # 输入层：卷积层，使用ReLU激活函数
        layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
        # 池化层：最大池化层
        layers.MaxPooling2D(),
        # 第二个卷积层
        layers.Conv2D(64, 3, activation='relu'),
        # 第二个池化层
        layers.MaxPooling2D(),
        # 第三个卷积层
        layers.Flatten(),
        # 全连接层：使用ReLU激活函数
        layers.Dense(128, activation='relu'),
        # 输出层：10个神经元，使用softmax激活函数进行多分类
        layers.Dense(10, activation='softmax'),
        # Dropout层：防止过拟合
        layers.Dropout(0.15)
    ])
    # 编译模型：使用Adam优化器，损失函数为稀疏分类交叉熵，评估指标为准确率
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 训练模型并记录训练过程
    history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))

    # 保存模型
    model.save(MODEL_PATH)
    print(f'模型已保存到 {MODEL_PATH}')

    # 保存训练过程数据到csv
    import pandas as pd
    df = pd.DataFrame(history.history)
    df.to_csv('train_history.csv', index=False)
    print('训练过程数据已保存到 train_history.csv')

    # 保存损失曲线和准确率曲线为图片
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('train_curves.png')
    print('训练曲线已保存为 train_curves.png')
    plt.show()
    # 保存部分预测结果图像
    fig = plt.figure(figsize=(10,4))
    idx = np.random.choice(len(x_test), 10, replace=False)
    images = x_test[idx]
    labels = y_test[idx]
    preds = np.argmax(model.predict(images), axis=1)
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'True: {labels[i]}\nPred: {preds[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('predict_samples.png')
    print('部分预测结果已保存为 predict_samples.png')
    plt.show()

if __name__ == '__main__':
    train_and_save_model()
