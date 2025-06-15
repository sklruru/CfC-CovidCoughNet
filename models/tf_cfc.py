import tensorflow as tf
import numpy as np


# LeCun improved tanh activation
# http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
def lecun_tanh(x):
    return 1.7159 * tf.nn.tanh(0.666 * x)


class CfcCell(tf.keras.layers.Layer):
    """
    Closed-form Continuous-time Neural Network (CfC) cell implementation
    参考原始论文: https://www.nature.com/articles/s42256-022-00556-7
    """

    def __init__(self, units, hparams=None, **kwargs):
        super(CfcCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units

        # 默认超参数
        default_hparams = {
            "backbone_activation": "silu",
            "backbone_layers": 1,
            "backbone_units": units,
            "backbone_dr": 0.1,
            "weight_decay": 1e-6,
        }

        self.hparams = default_hparams
        if hparams is not None:
            for k, v in hparams.items():
                self.hparams[k] = v

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        # 设置激活函数
        if self.hparams.get("backbone_activation") == "silu":
            backbone_activation = tf.nn.silu
        elif self.hparams.get("backbone_activation") == "relu":
            backbone_activation = tf.nn.relu
        elif self.hparams.get("backbone_activation") == "tanh":
            backbone_activation = tf.nn.tanh
        elif self.hparams.get("backbone_activation") == "gelu":
            backbone_activation = tf.nn.gelu
        elif self.hparams.get("backbone_activation") == "lecun":
            backbone_activation = lecun_tanh
        elif self.hparams.get("backbone_activation") == "softplus":
            backbone_activation = tf.nn.softplus
        else:
            raise ValueError("Unknown backbone activation")

        # 配置选项
        self._no_gate = False
        if "no_gate" in self.hparams:
            self._no_gate = self.hparams["no_gate"]
        self._minimal = False
        if "minimal" in self.hparams:
            self._minimal = self.hparams["minimal"]

        # 构建骨干网络（特征提取器）
        self.backbone = []
        for i in range(self.hparams["backbone_layers"]):
            self.backbone.append(
                tf.keras.layers.Dense(
                    self.hparams["backbone_units"],
                    backbone_activation,
                    kernel_regularizer=tf.keras.regularizers.L2(
                        self.hparams["weight_decay"]
                    ),
                )
            )
            self.backbone.append(tf.keras.layers.Dropout(self.hparams["backbone_dr"]))

        self.backbone = tf.keras.models.Sequential(self.backbone)

        # 构建CFC特定层
        if self._minimal:
            # 简化版CFC
            self.ff1 = tf.keras.layers.Dense(
                self.units,
                kernel_regularizer=tf.keras.regularizers.L2(
                    self.hparams["weight_decay"]
                ),
            )
            self.w_tau = self.add_weight(
                shape=(1, self.units), initializer=tf.keras.initializers.Zeros()
            )
            self.A = self.add_weight(
                shape=(1, self.units), initializer=tf.keras.initializers.Ones()
            )
        else:
            # 完整版CFC
            self.ff1 = tf.keras.layers.Dense(
                self.units,
                lecun_tanh,
                kernel_regularizer=tf.keras.regularizers.L2(
                    self.hparams["weight_decay"]
                ),
            )
            self.ff2 = tf.keras.layers.Dense(
                self.units,
                lecun_tanh,
                kernel_regularizer=tf.keras.regularizers.L2(
                    self.hparams["weight_decay"]
                ),
            )
            self.time_a = tf.keras.layers.Dense(
                self.units,
                kernel_regularizer=tf.keras.regularizers.L2(
                    self.hparams["weight_decay"]
                ),
            )
            self.time_b = tf.keras.layers.Dense(
                self.units,
                kernel_regularizer=tf.keras.regularizers.L2(
                    self.hparams["weight_decay"]
                ),
            )
        self.built = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "hparams": self.hparams,
        })
        return config

    def call(self, inputs, states, **kwargs):
        hidden_state = states[0]
        t = 1.0  # 默认时间步长为1

        # 处理输入可能是tuple或list的情况
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            t = tf.reshape(elapsed, [-1, 1])
            inputs = inputs[0]

        # 连接输入和隐藏状态
        x = tf.keras.layers.Concatenate()([inputs, hidden_state])
        x = self.backbone(x)
        ff1 = self.ff1(x)

        if self._minimal:
            # 简化版CFC解决方案
            new_hidden = (
                    -self.A
                    * tf.math.exp(-t * (tf.math.abs(self.w_tau) + tf.math.abs(ff1)))
                    * ff1
                    + self.A
            )
        else:
            # 标准CFC
            ff2 = self.ff2(x)
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = tf.nn.sigmoid(-t_a * t + t_b)

            if self._no_gate:
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden, [new_hidden]


class CfC(tf.keras.Model):
    """
    基于CfC的模型，用于时间序列分类任务
    """

    def __init__(self, units, hparams=None, return_sequences=False, output_units=2):
        super(CfC, self).__init__()

        # 设置默认超参数
        default_hparams = {
            "backbone_activation": "silu",
            "backbone_layers": 1,
            "backbone_units": units,
            "backbone_dr": 0.1,
            "weight_decay": 1e-6,
        }

        self.hparams = default_hparams
        if hparams is not None:
            for k, v in hparams.items():
                self.hparams[k] = v

        self.units = units
        self.return_sequences = return_sequences
        self.output_units = output_units
        self.cell = CfcCell(units, self.hparams)
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=return_sequences)
        self.output_layer = tf.keras.layers.Dense(output_units, activation="softmax")

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "hparams": self.hparams,
            "return_sequences": self.return_sequences,
            "output_units": self.output_units,
        })
        return config

    def call(self, inputs, training=False):
        """前向传播函数"""
        x = self.rnn(inputs)
        return self.output_layer(x)


def build_cfc_model(input_shape, hparams=None, num_classes=2):
    """
    构建用于咳嗽音频分类的CFC模型

    Args:
        input_shape: 输入特征的形状 (时间步长, 特征维度)
        hparams: 超参数字典
        num_classes: 分类数量

    Returns:
        构建好的模型
    """
    # 设置超参数
    if hparams is None:
        hparams = {
            "backbone_activation": "silu",  # 激活函数: silu, relu, tanh, gelu, lecun, softplus
            "backbone_layers": 2,  # 骨干网络层数
            "backbone_units": 128,  # 骨干网络单元数
            "backbone_dr": 0.3,  # Dropout率
            "weight_decay": 1e-5,  # 权重衰减
            "minimal": False,  # 是否使用简化版CFC
            "no_gate": False,  # 是否禁用门控机制
        }

    # 构建模型
    model = tf.keras.Sequential()

    # 输入重塑 - 将特征转换为序列形式 (batch, time, features)
    # 如果输入不是3D的，先将其重塑
    if len(input_shape) == 3:
        # 输入形状是 (height, width, channels)
        model.add(tf.keras.layers.Reshape((input_shape[0], input_shape[1] * input_shape[2]), input_shape=input_shape))
    elif len(input_shape) == 2:
        # 输入形状是 (height, width)
        model.add(tf.keras.layers.Reshape((input_shape[0], input_shape[1]), input_shape=input_shape))

    # 第一个CFC层
    model.add(tf.keras.layers.RNN(
        CfcCell(128, hparams),
        return_sequences=True,
        name="cfc_1"
    ))
    model.add(tf.keras.layers.BatchNormalization())

    # 第二个CFC层
    model.add(tf.keras.layers.RNN(
        CfcCell(64, hparams),
        return_sequences=False,
        name="cfc_2"
    ))
    model.add(tf.keras.layers.BatchNormalization())

    # 全连接层进行特征融合和分类
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model