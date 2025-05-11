"""
花卉识别系统的模型训练脚本
用于读取花卉数据集，使用MobileNetV2进行迁移学习训练花卉分类模型，
评估模型效果并保存训练结果和可视化
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# 设置GPU内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 数据路径 - 花卉文件夹现在直接在项目根目录
data_dir = '.'  # 修改为当前目录

# 图像尺寸
img_height, img_width = 224, 224
batch_size = 32

# 数据增强和预处理
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 训练集
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],  # 明确指定类别
    subset='training'
)

# 验证集
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],  # 明确指定类别
    subset='validation'
)

# 类别名称
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # 直接定义类别名称
print(f"类别: {class_names}")
num_classes = len(class_names)

# 加载预训练的MobileNetV2模型
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# 冻结基础模型层
for layer in base_model.layers:
    layer.trainable = False

# 构建模型
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 打印模型概要
model.summary()

# 回调函数
checkpoint_path = "flower_model.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 训练模型
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# 保存模型
model.save('flower_model.h5')
print("模型已保存到 flower_model.h5")

# 绘制训练历史
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('模型准确率')
plt.ylabel('准确率')
plt.xlabel('轮次')
plt.legend(['训练集', '验证集'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('模型损失')
plt.ylabel('损失')
plt.xlabel('轮次')
plt.legend(['训练集', '验证集'], loc='upper right')
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# 评估模型
print("\n开始评估模型...")
test_loss, test_acc = model.evaluate(validation_generator)
print(f'测试准确率: {test_acc:.4f}')

# 预测验证集
validation_generator.reset()
y_pred = []
y_true = []

for i in range(len(validation_generator)):
    x, y = validation_generator[i]
    predictions = model.predict(x)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(np.argmax(y, axis=1))
    
    if len(y_true) >= validation_generator.samples:
        break

# 截取到验证集样本数量
y_true = y_true[:validation_generator.samples]
y_pred = y_pred[:validation_generator.samples]

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# 输出分类报告
print("\n分类报告:")
print(classification_report(y_true, y_pred, target_names=class_names)) 