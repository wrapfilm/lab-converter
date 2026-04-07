from flask import Flask, render_template, request, jsonify
import numpy as np
from datetime import datetime, timedelta, timezone

app = Flask(__name__)

# 配置：D65/10° 白点
XN, YN, ZN = 94.811, 100.000, 107.304
CHINA_TZ = timezone(timedelta(hours=8))


@app.context_processor
def inject_now():
    now_china = datetime.now(timezone.utc).astimezone(CHINA_TZ)
    return {'current_year': now_china.year}


# --- 核心算法函数 ---

def f(t):
    """XYZ to Lab 的辅助函数"""
    return t ** (1 / 3) if t > 0.008856 else (903.3 * t + 16) / 116


def f_inv(t):
    """Lab to XYZ 的辅助函数"""
    return t ** 3 if t > (6 / 29) else (116 * t - 16) / 903.3


def apply_gamma_srgb(linear):
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)


def remove_gamma_srgb(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4))


# 转换矩阵 (D65)
M_ADOBE = np.array([
    [2.0413690, -0.5649464, -0.3446944],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0134474, -0.1183897, 1.0154096]
])

M_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]
])


# --- 路由逻辑 ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/color_space_guide')
def color_space_guide():
    # 指向你刚才放入 templates 文件夹的 guide.html
    return render_template('guide.html')

@app.route('/convert', methods=['POST'])
def convert():
    data = request.json
    mode = data.get('mode')

    try:
        if mode == 'lab2rgb':
            l, a, b = float(data['l']), float(data['a']), float(data['b'])
            # Lab -> XYZ
            fy = (l + 16) / 116
            fx, fz = a / 500 + fy, fy - b / 200
            xyz = np.array([f_inv(fx) * XN, f_inv(fy) * YN, f_inv(fz) * ZN]) / 100.0

            # Adobe RGB
            lin_a = np.clip(M_ADOBE.dot(xyz), 0, 1)
            rgb_a = (np.power(lin_a, 1 / 2.2) * 255).round().astype(int)

            # sRGB
            lin_s = np.clip(M_SRGB.dot(xyz), 0, 1)
            rgb_s = (apply_gamma_srgb(lin_s) * 255).round().astype(int)

            return jsonify({'success': True, 'adobe_rgb': rgb_a.tolist(), 's_rgb': rgb_s.tolist()})

        elif mode == 'rgb2lab':
            r, g, b = int(data['r']), int(data['g']), int(data['b'])
            space = data.get('space')  # 'srgb' or 'adobe'
            rgb_norm = np.array([r, g, b]) / 255.0

            if space == 'adobe':
                lin = np.power(rgb_norm, 2.2)
                xyz = np.linalg.inv(M_ADOBE).dot(lin) * 100.0
            else:
                lin = remove_gamma_srgb(rgb_norm)
                xyz = np.linalg.inv(M_SRGB).dot(lin) * 100.0

            # XYZ -> Lab
            vx, vy, vz = f(xyz[0] / XN), f(xyz[1] / YN), f(xyz[2] / ZN)
            lab = [round(116 * vy - 16, 2), round(500 * (vx - vy), 2), round(200 * (vy - vz), 2)]
            return jsonify({'success': True, 'lab': lab})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)