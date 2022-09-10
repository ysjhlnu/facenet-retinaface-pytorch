from crypt import methods
from flask import Flask, jsonify, request
import time
import numpy as np
import cv2
import os
import random
import base64, string
from retinaface import Retinaface

app = Flask(__name__)
retinaface = Retinaface()


# path2base64 图片文件转为base64格式
def path2base64(path):
    with open(path, "rb") as f:
        byte_data = f.read()
    base64_str = base64.b64encode(byte_data).decode("ascii")  # 二进制转base64
    return base64_str


@app.route("/infer", methods=["POST"])
def predict():
    result = {"success": False}
    if request.method == "POST":
        if request.files.get("image") is not None:
            try:
                # 得到客户端传输的图像          
                start = time.time()      
                input_image = request.files["image"].read()
                imBytes = np.frombuffer(input_image, np.uint8)
                iImage = cv2.imdecode(imBytes, cv2.IMREAD_COLOR)
                # 执行推理
                r_image, label_result = retinaface.detect(iImage)
                print("duration: ",time.time()-start)
 
                if (label_result is None) and (len(label_result) < 0):
                    result["success"] = False
                # 将结果保存为json格式
                print(label_result)
                result["box"] = label_result
                result['success'] = True
            except Exception as e:
                print(str(e))
    
    return jsonify(result)


@app.route("/", methods=["POST"])
def default():
        print(request.form.get("fpath"))
        print(request.form.get("fname"))   # 图片名称
        print(request.form.get("ftype"))
        img_type = request.form.get("img_type")        # 上传的图片数据类型
        data = request.get_json()
        f = data['fdata']  # 图片数据
        img_path = ''.join(random.sample(string.ascii_letters + string.digits, 12)) + '.jpg'
        img_data = base64.b64decode(f)

        file = open(img_path, 'wb')
        file.write(img_data)
        file.close()
        
        iImage = cv2.imread(img_path)
        image   = cv2.cvtColor(iImage,cv2.COLOR_BGR2RGB)
        r_image, label_result = retinaface.detect(image)
        r_image = cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR)
        save_path = ''.join(random.sample(string.ascii_letters + string.digits, 12)) + '.jpg'
        cv2.imwrite(save_path, r_image)
        result = {}
        result_bytes = path2base64(save_path)
                
        result["box"] = label_result
        result["IndexNum"] = ""
        result["Index"] = len(label_result)
        result["base64"] = result_bytes

        os.remove(img_path)
        os.remove(save_path)
        
        return jsonify(result)


@app.route("/add_face", methods=['POST'])
def add_face():
    base_dir = './face_dataset/'
    
    img_name = request.form.get("img_name")
    if img_name is None:
        return jsonify({"error":"图片名称不存在", "code": -1})
    
    id = 1
    save_path = ""
    while True:
        save_path = base_dir + img_name + '_%d.jpg' % id
        if not os.path.exists(save_path):
            break
        id += 1
    print(save_path)
    
    if request.files.get("image") is not None:
            input_image = request.files["image"].read()
            imBytes = np.frombuffer(input_image, np.uint8)
            iImage = cv2.imdecode(imBytes, cv2.IMREAD_COLOR)
            cv2.imwrite(save_path, iImage)
    else:
        img_name = request.form.get("img_name")        # 上传的图片数据类型
        data = request.get_json()
        f = data['image']

        img_data = base64.b64decode(f)
        file = open(save_path, 'wb')
        file.write(img_data)
        file.close()
        
    return jsonify({"code": 0, "msg": "添加成功"})
    
    
@app.route("/reload_face_lib", methods=['POST'])
def reload_facenet_lib():
    retinaface.reload_face_lib();
    return jsonify({"code": 0, "msg": "加载成功"})


if __name__ == "__main__":
    print(("* Flask starting server..."
        "please wait until server has fully started"))
    app.run(host='0.0.0.0', port=7001)