import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
import sys
import os
import cv2
# 定义模型保存路径
# 如果在同一目录下运行，确保train_model.py和mnist_cnn_model
# 否则需要修改路径
MODEL_PATH = 'mnist_cnn_model.keras'

# 如果模型文件不存在，将会调用train_model.py进行训练
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        # 如果模型文件存在，直接加载
        model = keras.models.load_model(MODEL_PATH)
        print(f'已加载模型: {MODEL_PATH}')
    else:
        # 如果模型文件不存在，调用训练脚本
        import subprocess
        print('未检测到模型文件，正在训练模型...')
        # 使用sys.executable确保使用当前Python环境运行脚本
        # 如果在Windows上运行，可能需要使用train_model.exe
        subprocess.run(['train_model.exe'])
        model = keras.models.load_model(MODEL_PATH)
    # 返回加载或训练好的模型
    print('模型加载完成')
    return model


# 6. 用户交互：图片识别
# 对用户上传的图片进行预处理，适配模型输入
def preprocess_user_image(img, show=False):
    # 输入img为PIL.Image对象，转为28x28灰度图并二值化、居中
    # 如果需要显示预处理后的图片，可以设置show=True

    # 将图片转换为灰度图并调整大小
    img = img.convert('L').resize((28, 28))
    # 将PIL.Image转换为NumPy数组
    # 这里使用numpy.array()将PIL.Image转换为NumPy数组
    arr = np.array(img)
    # 使用Otsu's方法进行二值化
    # 这里使用skimage.filters.threshold_otsu()计算Otsu阈值  
    from skimage.filters import threshold_otsu
    # 计算Otsu阈值
    thresh = threshold_otsu(arr)
    # 将数组转换为二值图像
    # 这里使用NumPy的布尔索引将数组转换为二值图
    binary = (arr > thresh).astype(np.uint8) * 255
    # 如果二值化后白色像素多于黑色像素，反转颜色
    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = 255 - binary
    # 居中处理
    # 使用scipy.ndimage.center_of_mass()计算二值图像的质心
    from scipy.ndimage import center_of_mass, shift
    cy, cx = center_of_mass(binary < 128)
    shift_y, shift_x = np.array(binary.shape) // 2 - np.array([cy, cx])
    shifted = shift(binary, shift=(shift_y, shift_x), cval=0)
    
    # 使用OpenCV的resize函数将图像缩放到28x28
    
    norm = cv2.resize(shifted.astype('uint8'), (28, 28), interpolation=cv2.INTER_AREA)
    # 将图像数据转换为浮点数并归一化到0-1
    norm = shifted.astype('float32') / 255.0
    # 扩展维度以适应模型输入
    # 这里使用NumPy的expand_dims函数将数组的形状扩展为
    # (1, 28, 28, 1)，适应Keras模型的输入要求
    norm = np.expand_dims(norm, axis=(0, -1))
    if show:
        # 如果需要显示预处理后的图片，可以使用matplotlib.pyplot.imshow()
        plt.imshow(norm[0, :, :, 0], cmap='gray')
        plt.title('Preprocessed Image')
        plt.axis('off')
        plt.show()
    # 返回预处理后的图像数据
    # 返回的norm是一个形状为(1, 28, 28, 1)的NumPy数组，
    # 适合输入到Keras模型中进行预测
    return norm

# 对用户图片进行数字分割，返回每个数字的PIL.Image对象列表
def segment_digits(image):
    # 输入image为PIL.Image对象，返回每个数字的PIL.Image对象列表
    # 将图片转换为灰度图并二值化
    img = np.array(image.convert('L'))
    # 使用Otsu's方法进行二值化
    # 这里使用skimage.filters.threshold_otsu()计算Otsu阈值
    from skimage.filters import threshold_otsu
    # 计算Otsu阈值
    thresh = threshold_otsu(img)
    # 将数组转换为二值图像
    binary = (img > thresh).astype(np.uint8) * 255
    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = 255 - binary
    # 轮廓检测，找到所有数字区域
    # 使用OpenCV的findContours函数找到所有轮廓, 
    # cv2.RETR_EXTERNAL表示只检测外部轮廓，cv2.CHAIN_APPROX_SIMPLE表示压缩水平、垂直和对角线段
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 对每个轮廓进行处理，提取数字区域
    # 并将其转换为28x28的PIL.Image对象
    digit_imgs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:
            # 扩大边界，保留部分背景
            pad = int(0.2 * max(w, h))
            x0 = max(x - pad, 0)
            y0 = max(y - pad, 0)
            x1 = min(x + w + pad, binary.shape[1])
            y1 = min(y + h + pad, binary.shape[0])
            digit = binary[y0:y1, x0:x1]
            # 居中到28x28
            square = np.zeros((max(digit.shape), max(digit.shape)), dtype=np.uint8)
            y_offset = (square.shape[0] - digit.shape[0]) // 2
            x_offset = (square.shape[1] - digit.shape[1]) // 2
            square[y_offset:y_offset+digit.shape[0], x_offset:x_offset+digit.shape[1]] = digit
            try:
                digit_img = Image.fromarray(square).resize((28, 28), Image.Resampling.LANCZOS)
            except AttributeError:
                digit_img = Image.fromarray(square).resize((28, 28), Image.ANTIALIAS)
            digit_imgs.append((x, digit_img))
    # 按照x坐标排序，确保从左到右的顺序
    # 这里使用Python的内置sorted函数和lambda表达式对列表进行排序
    digit_imgs.sort(key=lambda tup: tup[0])
    
    return [img for _, img in digit_imgs]

# 预测图片中的数字，只取第一个分割出的数字
def predict_image(img_path, model, show=False):
    img = Image.open(img_path)
    digits = segment_digits(img)
    if not digits:
        return None
    # 只识别第一个数字（通常为最左侧）
    digit_img = digits[0]
    arr = preprocess_user_image(digit_img, show=False)
    pred = int(np.argmax(model.predict(arr), axis=1)[0])
    return pred

# 图形界面，支持用户选择图片并显示识别结果
def gui_predict():
    model = load_or_train_model()
    def select_image():
        file_path = filedialog.askopenfilename(filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.bmp')])
        if file_path:
            img = Image.open(file_path)
            # 放大显示原图
            display_img = img.resize((280, 280))
            tk_img = ImageTk.PhotoImage(display_img)
            img_label.config(image=tk_img)
            img_label.image = tk_img
            pred = predict_image(file_path, model=model, show=False)
            result_var.set(f'识别结果: {pred}')
    root = tk.Tk()
    root.title('手写数字识别(Keras)')
    root.geometry('600x500')
    root.configure(bg='#f0f0f0')
    title_label = tk.Label(root, text='手写数字识别', font=('Arial', 24, 'bold'), bg='#f0f0f0')
    title_label.pack(pady=20)
    btn = tk.Button(root, text='选择图片', command=select_image, font=('Arial', 14), width=15, bg='#4a90e2', fg='white')
    btn.pack(pady=10)
    img_label = tk.Label(root, bg='#d9d9d9', width=280, height=280)
    img_label.pack(pady=20)
    result_var = tk.StringVar()
    result_label = tk.Label(root, textvariable=result_var, font=('Arial', 20), bg='#f0f0f0', fg='#333')
    result_label.pack(pady=10)
    root.mainloop()

# 程序入口，支持命令行图片识别和GUI两种方式
if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 如果命令行参数中有图片路径，进行图片识别
        model = load_or_train_model()
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f'错误: 图片文件 {img_path} 不存在')
            sys.exit(1)
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print('错误: 只支持PNG、JPG、JPEG和BMP格式的图片')
            sys.exit(1)
        # 调用预测函数
        pred = predict_image(img_path, model=model)
        print(f'图片中的数字预测为: {pred}')
    else:
        gui_predict()
