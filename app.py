"""
花卉识别系统的Streamlit Web应用界面
提供简洁的用户界面，用于上传图片进行花卉分类预测，显示预测结果和概率
"""
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

# 页面配置
st.set_page_config(
    page_title="花卉识别系统",
    page_icon="🌼",
    layout="centered"
)

# 标题
st.title("花卉识别系统 🌷🌻🌹🌼🌺")
st.markdown("上传一张花卉图片，AI 将识别它属于哪一类花卉")

# 加载模型
@st.cache_resource
def load_flower_model():
    try:
        model = load_model('flower_model.h5')
        return model
    except:
        st.error("模型文件未找到，请先运行 train.py 训练模型！")
        return None

# 获取类别名称 - 按照模型训练时的顺序
flower_classes = ['雏菊', '蒲公英', '玫瑰', '向日葵', '郁金香']
folder_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # 对应的文件夹名称

# 图像预处理
def preprocess_image(img):
    # 转换为RGB模式，确保只有3个通道
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# 预测函数
def predict_flower(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    return predicted_class, confidence, predictions[0]

# 加载模型
model = load_flower_model()

# 文件上传器
uploaded_file = st.file_uploader("请选择一张花卉图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示上传的图片
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="上传的图片", width=300)
    
    # 添加预测按钮
    if st.button("开始识别"):
        with st.spinner('AI正在分析图片...'):
            # 预处理图像
            processed_image = preprocess_image(Image.open(uploaded_file))
            
            # 预测
            if model:
                predicted_class, confidence, all_predictions = predict_flower(model, processed_image)
                
                # 显示结果
                st.success(f"识别结果: **{flower_classes[predicted_class]}**")
                st.info(f"置信度: {confidence:.2%}")
                
                # 显示所有类别的预测概率
                st.subheader("各类别预测概率")
                for i, (flower, prob) in enumerate(zip(flower_classes, all_predictions)):
                    st.progress(float(prob))
                    st.text(f"{flower}: {prob:.2%}")
                    
                # 分隔线
                st.markdown("---")
                
                # 添加额外信息
                if predicted_class == 0:  # 雏菊
                    st.markdown("**雏菊 (Daisy)**: 雏菊花朵小巧可爱，花瓣白色，花心黄色，常见于草地上。")
                elif predicted_class == 1:  # 蒲公英
                    st.markdown("**蒲公英 (Dandelion)**: 蒲公英有明亮的黄色花朵，成熟后变为蓬松的白色种子头。")
                elif predicted_class == 2:  # 玫瑰
                    st.markdown("**玫瑰 (Rose)**: 玫瑰是爱情的象征，有各种颜色，花瓣层叠丰富。")
                elif predicted_class == 3:  # 向日葵
                    st.markdown("**向日葵 (Sunflower)**: 向日葵有大而明亮的黄色花朵，花盘较大，总是朝向太阳。")
                elif predicted_class == 4:  # 郁金香
                    st.markdown("**郁金香 (Tulip)**: 郁金香花形优雅，有各种鲜艳的颜色，花瓣光滑且有光泽。")

# 侧边栏信息
with st.sidebar:
    st.header("关于")
    st.info(
        """
        本应用使用深度学习识别花卉图片。
        
        **支持的花卉类别:**
        - 雏菊 (Daisy)
        - 蒲公英 (Dandelion)
        - 玫瑰 (Rose)
        - 向日葵 (Sunflower)
        - 郁金香 (Tulip)
        
        模型基于MobileNetV2架构，通过迁移学习训练。
        """
    )
    
    st.header("使用说明")
    st.markdown(
        """
        1. 点击"选择一张花卉图片"上传图片
        2. 点击"开始识别"按钮
        3. 查看识别结果和预测概率
        """
    ) 