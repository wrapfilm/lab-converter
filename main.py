from flask import Flask, render_template, request, jsonify
import numpy as np
from datetime import datetime

# 这个装饰器会让 'current_year' 在所有模板中都可用

app = Flask(__name__)

# 标准 D65 白点
XN, YN, ZN = 95.047, 100.000, 108.883

@app.context_processor
def inject_now():
    return {'current_year': datetime.utcnow().year}



def f_inv(t):
    return t ** 3 if t ** 3 > 0.008856 else (116 * t - 16) / 903.3


def f(t):
    return t ** (1 / 3) if t > 0.008856 else (903.3 * t + 16) / 116


def lab_to_adobe_rgb(l, a, b):
    # 1. Lab -> XYZ
    fy = (l + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    x, y, z = f_inv(fx) * XN, f_inv(fy) * YN, f_inv(fz) * ZN

    # 2. XYZ -> Linear Adobe RGB
    xyz = np.array([x, y, z]) / 100.0
    matrix = np.array([
        [2.04136, -0.56495, -0.34469],
        [-0.96927, 1.87601, 0.04156],
        [0.01345, -0.11839, 1.01541]
    ])
    rgb_linear = matrix.dot(xyz)
    rgb_linear = np.clip(rgb_linear, 0, 1)

    # 3. Gamma 校正
    rgb = np.power(rgb_linear, 1 / 2.2)
    return (rgb * 255).astype(int).tolist()


def adobe_rgb_to_lab(r, g, b):
    # 1. 反向 Gamma 校正
    rgb = np.array([r, g, b]) / 255.0
    rgb_linear = np.power(rgb, 2.2)

    # 2. Linear Adobe RGB -> XYZ
    # 逆矩阵
    inv_matrix = np.linalg.inv(np.array([
        [2.04136, -0.56495, -0.34469],
        [-0.96927, 1.87601, 0.04156],
        [0.01345, -0.11839, 1.01541]
    ]))
    xyz = inv_matrix.dot(rgb_linear) * 100.0

    # 3. XYZ -> Lab
    vx, vy, vz = f(xyz[0] / XN), f(xyz[1] / YN), f(xyz[2] / ZN)

    l = 116 * vy - 16
    a = 500 * (vx - vy)
    b = 200 * (vy - vz)
    return [round(l, 2), round(a, 2), round(b, 2)]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/convert', methods=['POST'])
def convert():
    data = request.json
    mode = data.get('mode')  # 'lab2rgb' 或 'rgb2lab'
    try:
        if mode == 'lab2rgb':
            res = lab_to_adobe_rgb(float(data['l']), float(data['a']), float(data['b']))
            return jsonify({'success': True, 'result': res, 'type': 'rgb'})
        else:
            res = adobe_rgb_to_lab(int(data['r']), int(data['g']), int(data['b']))
            return jsonify({'success': True, 'result': res, 'type': 'lab'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)