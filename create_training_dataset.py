# -*- coding: utf-8 -*-
"""
Created on Oct 2025
generate the traces for training the Transformer model
@author: Jingyu

本脚本用于“批量合成”MS2 实验中的启动子状态序列（ON/OFF）与对应的荧光轨迹数据集用于模型训练：
1) 先用二状态马尔可夫链生成若干条 promoter 状态序列（0=OFF, 1=ON）。
2) 再用 MS2“卷积核”（窗口 W）把“过去 W 帧的状态”映射为“当前帧的期望荧光均值”，随后加高斯噪声得到观测值。
3) 保存 promoter 真值与合成荧光到 CSV/MAT 文件，并画几条轨迹以便快速质检。

术语速记：
- K=2：状态数（ON/OFF）。
- W：MS2 卷积的窗口长度；W 越大，“记忆”越长；复合状态数 ~ 2^W。
- kappa=t_MS2/deltaT：离散化后 MS2 标签长度（单位：帧数），用于生成卷积核权重。
"""
import numpy as np
import matplotlib.pyplot as plt
from burstInfer.get_adjusted import get_adjusted  # 给定复合状态与核系数 → 返回 (ON权重, OFF权重)
from burstInfer.ms2_loading_coeff import ms2_loading_coeff  # 由 kappa 和 W 生成 MS2 卷积系数
import os

# -------- 随机性控制：保存本次随机种子便于复现 --------
seed_setter = np.random.randint(0, 1000000000)  # 实际跑时随机；若想完全可复现，可改成常数
np.random.seed(seed_setter)

# -------- 输出目录：建议确保存在 --------
cwd = os.getcwd()
print("Current working directory:", cwd)
dataset_path = cwd + '/dataset/'
train_path = dataset_path + 'train/'
test_path = dataset_path + 'test/'
print("Dataset path:", dataset_path)
print("Train path:", train_path)
print("Test path:", test_path)

def generate_training_dataset(number_of_traces, length_of_each_trace, is_train):
    # 二状态转移矩阵参数（稳态情形下用到）
    k_off_off = 0.8810  # P(OFF->OFF)
    k_on_on = 0.8567  # P(ON->ON)

    #%% 生成 promoter 状态序列（马尔可夫链抽样）
    def generate_promotor(transition_probabilities, number_of_traces, lengh_of_each_trace, starting_value):
        """
        参数：
          transition_probabilities: 2x2 转移矩阵 A
             A[0,0]=P(OFF->OFF), A[0,1]=P(OFF->ON), A[1,0]=P(ON->OFF), A[1,1]=P(ON->ON)
          number_of_traces: 生成多少条序列
          length_of_each_trace: 每条序列的长度（帧数）
          starting_value: 初值(可为标量/数组)，最终会广播/裁剪为长度 number_of_traces 的一维数组

        返回：
          chain_matrix: (number_of_traces, length_of_each_trace) 的 0/1 矩阵
        """

        print("----------------------------------------------")
        print("Generating promoter states...")
        print("The transition matrix is:" + str(transition_probabilities))
        print("The number of traces is: " + str(number_of_traces))
        print("The length of each trace is: " + str(lengh_of_each_trace))
        print("The starting value is: " + str(starting_value))

        chain_matrix = np.ones((number_of_traces, lengh_of_each_trace))

        # 规范化 starting_value：允许标量或任意形状，最终变为长度 number_of_traces 的一维数组
        sv = np.asarray(starting_value)

        for j in range(number_of_traces):
            if j % 10000 == 0:
                print("Generating trace " + str(j) + " ...")
            chain_length = lengh_of_each_trace
            chain = np.zeros((chain_length))
            chain[0] = sv[j]
            for i in range(1, chain_length):
                # 依据前一时刻状态选择对应行的类别分布抽样下一时刻
                # 1) 取上一时刻状态
                prev = int(chain[i - 1])  # 0或1

                # 2) 选转移矩阵对应的一行作为“类别分布”
                this_step_distribution = transition_probabilities[prev]  # 形如 [p(prev->0), p(prev->1)]

                # 3) 构造累计概率（CDF），用于“逆变换抽样”
                cumulative_distribution = np.cumsum(this_step_distribution)  # 形如 [p0, p0+p1(=1)]

                # 4) 从 U(0,1) 均匀分布抽一个数 r
                r = np.random.rand()

                # 5) 找到“第一个使 CDF > r 的下标”作为本时刻的类别
                #    这一步等价于从二类分布里抽 {0,1}
                chain[i] = np.where(cumulative_distribution > r)[0][0]
            chain_matrix[j, :] = chain

        print("Done.")
        print("The shape of the generated promoter states is:" + str(chain_matrix.shape))
        print("-------------------------------------")
        return chain_matrix


    # -------------------- 生成 promoter 状态 --------------------
    transition_probabilities = np.zeros((2, 2))  # [OFF-OFF, OFF-ON; ON-OFF, ON-ON]
    transition_probabilities[0, 0] = k_off_off
    transition_probabilities[0, 1] = 1 - k_off_off
    transition_probabilities[1, 1] = k_on_on
    transition_probabilities[1, 0] = 1 - k_on_on

    # 初始状态全部设为 1（ON），当然也可以设为 0 或随机
    starting_value = np.ones(number_of_traces, dtype=int)
    chain_matrix = generate_promotor(transition_probabilities, number_of_traces, length_of_each_trace, starting_value)

    synthetic_x = np.arange(0, length_of_each_trace)
    # ---- 模型基础参数 ----
    K = 2  # 启动子状态数：ON/OFF
    W = 2  # MS2 卷积“记忆窗口”的长度（以帧为单位）

    # 发射模型参数（高斯）：注意“均值”并不是直接 mu[state]，
    # 而是 “ON均值 与 OFF均值 按 MS2 卷积权重做线性组合” 再加噪声。
    mu = np.zeros((K, 1))
    mu[0, 0] = 7096.5295359189  # OFF 状态的（基线）均值
    mu[1, 0] = 48700.3752       # ON 状态的（活跃）均值
    noise = 17364.9315703868    # 高斯噪声的标准差

    # kappa = t_MS2 / deltaT：MS2 标签延伸覆盖时长（秒）/ 成像帧间隔（秒）→ 以帧计的标签长度
    # 参考 GarciaLab cpHMM 文档中的定义（Table 1）
    t_MS2 = 30 # 聚合酶穿过 MS2 区域所需时间（秒）
    deltaT = 20 # 成像帧间隔（秒）
    kappa = t_MS2 / deltaT  # = 1.5

    # ---- 根据 kappa 与 W 生成离散 MS2 卷积核系数（长度= W） ----
    ms2_coeff = ms2_loading_coeff(kappa, W)

    # ---- 用位运算维护“滑动窗口的复合状态编码” ----
    # 以二进制编码最近 W 帧的 0/1（ON=1），如 W=2 时 mask=3（二进制 11）
    mask = np.int32((2 ** W) - 1)

    fluorescence_holder = np.zeros((number_of_traces, length_of_each_trace))

    for i in np.arange(0, len(fluorescence_holder)):
        if i % 10000 == 0:
            print("Generating fluorescence trace " + str(i) + " ...")
        single_promoter = np.expand_dims(chain_matrix[i, :], axis=0)  # (1, T) 的 0/1 序列
        single_trace = np.zeros((1, length_of_each_trace))

        # ----- 第 0 帧：窗口只有1位（历史为空），直接根据当前状态计算 -----
        t = 0
        window_storage = int(single_promoter[0, 0])  # 仅当前帧
        weights = get_adjusted(window_storage, K, W, ms2_coeff)  # 返回形如 [ON_weight, OFF_weight]
        # 期望均值 = ON_weight*mu_on + OFF_weight*mu_off；再加高斯噪声
        single_trace[0, t] = (weights[0] * mu[1, 0] + weights[1] * mu[0, 0]) + np.random.normal(0, noise)

        # ----- 从第 1 帧起：利用位运算维护“最近 W 帧”的编码 -----
        t = 1
        present_state_list = []
        # 注意：这里 append 的是“窗口编码”或“上一帧状态”，下面会左移并加入当前帧状态
        present_state_list.append(int(single_promoter[0, 0]))

        while t < length_of_each_trace:
            present_state = int(single_promoter[0, t])  # 当前帧 0/1

            # (present_state_list[t-1] << 1) + present_state : 在二进制上左移一位并放入当前帧
            # 与 mask 做按位与：确保只保留最近 W 位（窗口外的高位被清零）
            # 例：W=2 时，mask=0b11；若上一步=0b10，当前=1 → (0b10<<1)+1=0b101 → &0b11=0b01
            window_storage = np.bitwise_and((present_state_list[t - 1] << 1) + present_state, mask)

            present_state_list.append(window_storage)

            # 由复合状态编码 + 核系数 → 得到 (ON权重, OFF权重)
            weights = get_adjusted(window_storage, K, W, ms2_coeff)

            # 线性组合 ON/OFF 的均值，再加噪声得到观测
            single_trace[0, t] = (weights[0] * mu[1, 0] + weights[1] * mu[0, 0]) + np.random.normal(0, noise)

            t = t + 1

        fluorescence_holder[i, :] = single_trace

    print("Done.")
    print("The shape of fluorescence_holder is: " + str(fluorescence_holder.shape))

    # Plot a few randomly selected fluorescence traces and its promoter states for quick QC
    num_to_plot = 5
    random_indices = np.random.choice(number_of_traces, num_to_plot, replace=False)
    plt.figure(figsize=(12, 8))
    for idx, trace_idx in enumerate(random_indices):
        plt.subplot(num_to_plot, 1, idx + 1)
        plt.plot(synthetic_x, fluorescence_holder[trace_idx, :], label='Fluorescence Trace', color='blue')
        plt.step(synthetic_x, chain_matrix[trace_idx, :] * np.max(fluorescence_holder) * 0.8, where='post', label='Promoter State (scaled)', color='orange')
        plt.ylim([-1000, np.max(fluorescence_holder) * 1.1])
        plt.xlabel('Time (frames)')
        plt.ylabel('Fluorescence / Promoter State')
        plt.title(f'Trace {trace_idx}')
        plt.legend()
    plt.tight_layout()
    plt.show()

    # Save the generated promoter states and fluorescence traces into npy files
    if is_train:
        np.save(train_path + 'promoter_states.npy', chain_matrix)
        np.save(train_path + 'fluorescence_traces.npy', fluorescence_holder)
        print("Training dataset saved to .npy files.")
    else:
        np.save(test_path + 'promoter_states.npy', chain_matrix)
        np.save(test_path + 'fluorescence_traces.npy', fluorescence_holder)
        print("Testing dataset saved to .npy files.")

if __name__ == '__main__':
    # -------------------- 高层超参数 --------------------
    number_of_traces_training = 500000
    length_of_each_trace_training = 200
    number_of_traces_testing = 500000
    length_of_each_trace_testing = 200

    # Generate training dataset
    generate_training_dataset(number_of_traces_training, length_of_each_trace_training, is_train=True)
    # Generate testing dataset
    generate_training_dataset(number_of_traces_testing, length_of_each_trace_testing, is_train=False)
    print("All datasets generated.")
