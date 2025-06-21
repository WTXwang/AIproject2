import numpy as np
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import os
import time
from keras import Input, Model
# 定义模型保存路径
# 如果在同一目录下运行，确保train_model.py和mnist_cnn_model.keras在同一目录下
# 否则需要修改路径
MODEL_PATH = 'mnist_cnn_model.keras'

# 定义残差块
# 残差块是ResNet的核心组件，用于构建深层网络
# 残差块包含两个卷积层和一个捷径连接
# 捷径连接用于将输入直接添加到输出，帮助缓解梯度消失问题
# 残差块的输入和输出形状可以不同，如果形状不匹配，会通过1x1卷积调整形状
# 这里使用Keras的Functional API定义残差块
def residual_block(x, filters, kernel_size=3, stride=1):
    # 输入x为残差块的输入张量
    # filters为卷积层的输出通道数
    # kernel_size为卷积核大小，默认为3
    # stride为卷积步长，默认为1
    shortcut = x
    # 使用卷积层和批归一化层构建残差块
    # 第一个卷积层，使用ReLU激活函数
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    # 批归一化层，使用批归一化层对卷积层输出进行归一化
    x = layers.BatchNormalization()(x)
    # 第二个卷积层，不使用激活函数，保持输入形状
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    # 批归一化层，对第二个卷积层输出进行归一化
    x = layers.BatchNormalization()(x)

    # 如果步长不为1或输入和输出形状不匹配，使用1x1卷积调整形状
    # 这里使用1x1卷积和批归一化层调整捷径连接的形状
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    # 将残差块的输出和捷径连接相加
    # 使用Keras的add层将残差块输出和捷径连接相加
    x = layers.add([x, shortcut])
    # 这里使用Keras的Activation层对相加后的结果进行ReLU激活
    x = layers.Activation('relu')(x)
    # 返回残差块的输出张量
    return x

# 定义ResNet模型
# ResNet模型由多个残差块组成，适用于图像分类任务
def build_resnet(input_shape=(28,28,1), num_classes=10):
    # 输入形状为28x28x1的灰度图像，num_classes为分类数
    # 这里使用Keras的Input层定义输入张量
    inputs = Input(shape=input_shape)
    # 第一个卷积层，使用ReLU激活函数
    # 使用3x3卷积核，步长为1，填充方式为'same'
    x = layers.Conv2D(32, 3, strides=1, padding='same', activation='relu')(inputs)
    # 批归一化层，对卷积层输出进行归一化
    x = layers.BatchNormalization()(x)
    # 残差块1
    # 使用两个残差块，每个残差块包含两个卷积层
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    # 最大池化层，使用2x2池化核，步长为2，用于下采样，减少特征图的尺寸
    x = layers.MaxPooling2D()(x)
    # 残差块2
    # 使用两个残差块，每个残差块包含两个卷积层
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)

    # 全局平均池化层，将特征图的空间维度平均化，输出一个向量
    # 这里使用Keras的GlobalAveragePooling2D层对特征图进行全局平均池化
    x = layers.GlobalAveragePooling2D()(x)
    # 全连接层，使用ReLU激活函数
    # 这里使用Keras的Dense层定义全连接层，输出128个神经元
    x = layers.Dense(128, activation='relu')(x)

    # Dropout层，防止过拟合
    # 这里使用Keras的Dropout层，设置丢弃率为0.15
    x = layers.Dropout(0.15)(x)
    # 输出层，使用softmax激活函数进行多分类
    # 这里使用Keras的Dense层定义输出层，输出num_classes个神经元
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    # 使用Keras的Model类定义模型，输入为inputs，输出为outputs
    # 这里使用Keras的Model类将输入和输出张量连接起来，构建完整的ResNet模型
    model = Model(inputs, outputs)
    # 返回构建好的ResNet模型
    return model

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

    # 构建ResNet神经网络模型
    model = build_resnet(input_shape=(28,28,1), num_classes=10)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    
    start_time = time.time()
    # 训练模型并记录训练过程
    history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))
    end_time = time.time()
    
    
    # 创建输出目录，避免与其他模型冲突
    output_dir = 'output_resnet'
    os.makedirs(output_dir, exist_ok=True)

    # 保存训练时长到文件
    total_time = end_time - start_time
    print(f'训练总时长: {total_time:.2f} 秒')
    time_path = os.path.join(output_dir, 'train_time.txt')
    with open(time_path, 'w', encoding='utf-8') as f:
        f.write(f"训练总时长: {total_time:.2f} 秒\n")
    print(f'训练时长已保存到 {time_path}')

    # 保存模型
    model.save(MODEL_PATH)
    print(f'模型已保存到 {MODEL_PATH}')

    # 保存训练过程数据到csv
    import pandas as pd
    df = pd.DataFrame(history.history)
    csv_path = os.path.join(output_dir, 'train_history.csv')
    df.to_csv(csv_path, index=False)
    print(f'训练过程数据已保存到 {csv_path}')

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
    curve_path = os.path.join(output_dir, 'train_curves.png')
    plt.savefig(curve_path)
    print(f'训练曲线已保存为 {curve_path}')
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
    pred_path = os.path.join(output_dir, 'predict_samples.png')
    plt.savefig(pred_path)
    print(f'部分预测结果已保存为 {pred_path}')
    plt.show()

if __name__ == '__main__':
    train_and_save_model()
