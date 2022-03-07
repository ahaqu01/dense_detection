from flask import Flask, request, jsonify
from service_streamer import ThreadedStreamer

import ssl
# 在使用URLopen方法的时候，当目标网站使用的是自签名的证书时就会抛出这个错误
# 全局取消证书验证
from image_classifier import predict_image, batch_prediction

ssl._create_default_https_context = ssl._create_unverified_context

# 创建Flask app 和 模型
app = Flask(__name__)

# route
@app.route('/predict', methods=['POST'])
def predicted():
    if 'image' not in request.files:
        return jsonify({'error', 'Image not found'}),400
    if request.method == 'POST':
        image = request.files['image'].read() # 读取图片
        object_name = predict_image(image) # 预测结果
        return jsonify({'object_name':object_name}) #  jsonify 确保 response为 json格式


streamer = ThreadedStreamer(batch_prediction, batch_size=64) # 每次预测64张图片
@app.route('/stream_predict', methods=['POST'])
def stream_predict():
    if request.method == 'POST':
        image = request.files['image'].read() # 读取图片
        #print("streamer shape : ", streamer.predict([image]).shape)
        object_name = streamer.predict([image]*24)[0] # 预测输出
        return jsonify({'object_name':object_name})

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True,ssl_context=('/home/zhouyuhua/work/server.crt','/home/zhouyuhua/work/server.key'))
