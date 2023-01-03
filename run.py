# -*- coding: UTF-8 -*-
import numpy as np
import model
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/',methods=['POST','GET'])
def index():
    #  利用request取得使用者端傳來的方法為何
    if request.method == 'POST':
        #  利用request取得表單欄位值
        sepalLengthCm=request.values['sepalLengthCm']
        sepalWidthCm=request.values['sepalWidthCm']
        petalLengthCm=request.values['petalLengthCm']
        petalWidthCm=request.values['petalWidthCm']

        input = np.array([[float(sepalLengthCm),float(sepalWidthCm),float(petalLengthCm),float(petalWidthCm)]])
        result = model.predict(input)

        return render_template("index.html",input=input,result=result,
                               sepalLengthCm=sepalLengthCm,
                               sepalWidthCm=sepalWidthCm,
                               petalLengthCm=petalLengthCm,
                               petalWidthCm=petalWidthCm)

    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def postInput():
    # 取得前端傳過來的數值
    insertValues = request.get_json()
    x1=insertValues['sepalLengthCm']
    x2=insertValues['sepalWidthCm']
    x3=insertValues['petalLengthCm']
    x4=insertValues['petalWidthCm']
    input = np.array([[x1, x2, x3, x4]])
    result = model.predict(input)

    return jsonify({'return': str(result)})

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=3000, debug=False)