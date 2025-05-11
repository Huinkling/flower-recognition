"""
花卉识别系统的Gradio Web应用界面
提供交互式界面，用于上传图片进行花卉分类预测，展示预测结果和概率分布
"""
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# 加载模型
try:
    model = load_model('flower_model.h5')
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("请先运行 train.py 训练模型！")
    model = None

# 花卉类别 - 与train.py中保持一致
flower_classes = ['雏菊 (Daisy)', '蒲公英 (Dandelion)', '玫瑰 (Rose)', 
                  '向日葵 (Sunflower)', '郁金香 (Tulip)']

# 文件夹名称（用于参考）
folder_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# 花卉描述信息
flower_descriptions = {
    '雏菊 (Daisy)': "雏菊花朵小巧可爱，花瓣白色，花心黄色，常见于草地上。",
    '蒲公英 (Dandelion)': "蒲公英有明亮的黄色花朵，成熟后变为蓬松的白色种子头。",
    '玫瑰 (Rose)': "玫瑰是爱情的象征，有各种颜色，花瓣层叠丰富。",
    '向日葵 (Sunflower)': "向日葵有大而明亮的黄色花朵，花盘较大，总是朝向太阳。",
    '郁金香 (Tulip)': "郁金香花形优雅，有各种鲜艳的颜色，花瓣光滑且有光泽。"
}

# 图像预处理
def preprocess_image(img):
    # 将numpy数组转换为PIL图像，并确保为RGB格式（3通道）
    img = Image.fromarray(img).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# 创建条形图
def create_bar_chart(predictions):
    plt.figure(figsize=(10, 6))
    plt.bar(flower_classes, predictions)
    plt.title('各类别预测概率')
    plt.ylabel('概率')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 将图表转换为图像
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)

# 预测函数
def predict_flower(input_img):
    if model is None:
        return "请先运行 train.py 训练模型！", None, None
    
    # 预处理图像
    processed_img = preprocess_image(input_img)
    
    # 预测
    predictions = model.predict(processed_img)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_class = flower_classes[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    # 创建结果文本
    result_text = f"预测结果: {predicted_class}\n"
    result_text += f"置信度: {confidence:.2%}\n\n"
    result_text += f"描述: {flower_descriptions[predicted_class]}"
    
    # 创建条形图
    chart = create_bar_chart(predictions)
    
    return result_text, chart, {c: float(p) for c, p in zip(flower_classes, predictions)}

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 花卉识别系统 🌷🌻🌹🌼🌺")
    gr.Markdown("上传一张花卉图片，AI 将识别它属于哪一类花卉")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="上传花卉图片")
            predict_btn = gr.Button("开始识别", variant="primary")
        
        with gr.Column(scale=2):
            output_text = gr.Textbox(label="识别结果")
            output_chart = gr.Image(label="预测概率分布")
            output_confidence = gr.Label(label="各类别概率", num_top_classes=5)
    
    predict_btn.click(
        fn=predict_flower,
        inputs=input_image,
        outputs=[output_text, output_chart, output_confidence]
    )
    
    gr.Markdown("""
    ## 关于这个应用
    
    本应用使用深度学习识别花卉图片，基于迁移学习和MobileNetV2架构训练。
    
    **支持的花卉类别:**
    - 雏菊 (Daisy)
    - 蒲公英 (Dandelion)
    - 玫瑰 (Rose)
    - 向日葵 (Sunflower)
    - 郁金香 (Tulip)
    """)

    gr.Examples(
        examples=[
            "examples/daisy.jpg",
            "examples/dandelion.jpg",
            "examples/rose.jpg",
            "examples/sunflower.jpg",
            "examples/tulip.jpg"
        ],
        inputs=input_image,
        outputs=[output_text, output_chart, output_confidence],
        fn=predict_flower,
        cache_examples=True,
    )

# 启动应用
if __name__ == "__main__":
    demo.launch() 