# -*- coding: utf-8 -*-
"""
Created on Sep 2025
generate the traces
@author: Jon & Hongpeng & Jingyu

本脚本用于“合成”MS2 实验中的启动子状态序列（ON/OFF）与对应的荧光轨迹：
1) 先用二状态马尔可夫链生成若干条 promoter 状态序列（0=OFF, 1=ON）。
2) 再用 MS2“卷积核”（窗口 W）把“过去 W 帧的状态”映射为“当前帧的期望荧光均值”，随后加高斯噪声得到观测值。
3) 保存 promoter 真值与合成荧光到 CSV/MAT 文件，并画几条轨迹以便快速质检。

术语速记：
- K=2：状态数（ON/OFF）。
- W：MS2 卷积的窗口长度；W 越大，“记忆”越长；复合状态数 ~ 2^W。
- kappa=t_MS2/deltaT：离散化后 MS2 标签长度（单位：帧数），用于生成卷积核权重。
"""
import numpy as np
from scipy.io import savemat
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
from burstInfer.get_adjusted import get_adjusted       # 给定复合状态与核系数 → 返回 (ON权重, OFF权重)
from burstInfer.ms2_loading_coeff import ms2_loading_coeff  # 由 kappa 和 W 生成 MS2 卷积系数
import os

# -------- 随机性控制：保存本次随机种子便于复现 --------
seed_setter = np.random.randint(0, 1000000000)  # 实际跑时随机；若想完全可复现，可改成常数
np.random.seed(seed_setter)

# -------- 输出目录：建议确保存在 --------
cwd = os.getcwd()
parent = os.path.dirname(cwd)
data_folder_head = parent + '/' + 'Python code for data generation translate by Hongpeng/'
os.makedirs(data_folder_head, exist_ok=True)  # ✅ 防止目录不存在时报错

#%% 生成 promoter 状态序列（马尔可夫链抽样）
def generate_promotor(transition_probabilities, number_of_traces, lengh_of_each_trace, starting_value):
    """
    参数：
      transition_probabilities: 2x2 转移矩阵 A
         A[0,0]=P(OFF->OFF), A[0,1]=P(OFF->ON), A[1,0]=P(ON->OFF), A[1,1]=P(ON->ON)
      number_of_traces: 生成多少条序列
      lengh_of_each_trace: 每条序列的长度（帧数）
      starting_value: 初值(可为标量/数组)，最终会广播/裁剪为长度 number_of_traces 的一维数组

    返回：
      chain_matrix: (number_of_traces, lengh_of_each_trace) 的 0/1 矩阵
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
    # if sv.size == 0:
    #     # 空输入则默认全 0
    #     sv = np.zeros(number_of_traces, dtype=int)
    # elif sv.size == 1:
    #     # 单值广播
    #     sv = np.full(number_of_traces, int(np.squeeze(sv)), dtype=int)
    # else:
    #     # 尝试展平
    #     sv = sv.squeeze()
    #     if sv.ndim > 1:
    #         sv = sv.ravel()
    #     if sv.size < number_of_traces:
    #         # 长度不足则用最后一个值填充
    #         fill = np.full(number_of_traces - sv.size, int(sv[-1]), dtype=int)
    #         sv = np.concatenate([sv.astype(int, copy=False), fill])
    #     elif sv.size > number_of_traces:
    #         # 过长裁剪
    #         sv = sv.astype(int, copy=False)[:number_of_traces]
    #     else:
    #         sv = sv.astype(int, copy=False)

    for j in range(number_of_traces):
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
    print("The generated promoter states are:" + str(chain_matrix))
    print("-------------------------------------")
    return chain_matrix

# -------------------- 高层超参数 --------------------
onoff_dynamic_transition = False  # False=稳态转移(论文主设定)；True=分段动态转移(演示用)
number_of_traces = 500
lengh_of_each_trace = 200

# 若启用动态转移（True），将一条长序列划分为若干段，每段有不同的 k_on_on
number_of_segement_for_dynamic_traces = 20
segment_length = 10

# 二状态转移矩阵参数（稳态情形下用到）
k_off_off = 0.8810  # P(OFF->OFF)
k_on_on = 0.8567    # P(ON->ON)

# -------------------- 生成 promoter 状态 --------------------
if not onoff_dynamic_transition:
    k_on_array = k_on_on  # 仅用于保存到 MAT，稳态时这里是标量
    transition_probabilities = np.zeros((2, 2))  # [OFF-OFF, OFF-ON; ON-OFF, ON-ON]
    transition_probabilities[0, 0] = k_off_off
    transition_probabilities[0, 1] = 1 - k_off_off
    transition_probabilities[1, 1] = k_on_on
    transition_probabilities[1, 0] = 1 - k_on_on

    # 初始状态全部设为 1（ON），当然也可以设为 0 或随机
    starting_value = np.ones(number_of_traces, dtype=int)

    chain_matrix = generate_promotor(transition_probabilities, number_of_traces, lengh_of_each_trace, starting_value)
    promotor_csv_filename = data_folder_head + "synthetic_data_promotor" + '.csv'
else:

    # ====================== 分段动态转移 ======================
    # 思路：把整条轨迹分成 S 段（S = number_of_segement_for_dynamic_traces），
    # 每段长度为 L = segment_length。固定 P(OFF->OFF)=k_off_off，
    # 仅让 P(ON->ON) 随段号变化，从而在不同时间段模拟不同的“ON粘性/爆发持续性”。

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 生成随段号平滑变化的自变量 xdata：
    # 从 -2 到 2（不含 2），步长为 4/S，使得长度约为 S。
    # 注：np.arange 浮点步长可能受精度影响；如果担心边界问题，可改用 np.linspace(-2, 2, S, endpoint=False)
    xdata = np.arange(start=-2, stop=2, step=4 / number_of_segement_for_dynamic_traces, dtype='float')

    # k_on_array[i] = 当前段 i 的 P(ON->ON)，范围 (0,1)
    # 使用 sigmoid 让段间变化平滑（前期低、后期高，仅作示例）
    k_on_array = sigmoid(xdata)  # 形状约为 (S,)

    # 目标承载矩阵：存放最终拼好的所有 promoter 状态（0/1）
    # 形状：(条数 N, 总长度 S*L)
    chain_matrix = np.zeros((number_of_traces, number_of_segement_for_dynamic_traces * segment_length))

    # 下面两个索引用于把每段生成的序列“写回”到 chain_matrix 中的正确区间
    segment_save_index_end = segment_length  # 本段在大矩阵中的右边界（不含）
    segment_save_index_start = 0  # 本段在大矩阵中的左边界（含）

    # 段 1 的初始状态（每条轨迹的第 0 帧）；这里全部设为 1（ON），也可改为 0 或随机
    starting_value = np.ones(number_of_traces)  # 形状：(N,)

    # ---------- 主循环：逐段生成并拼接 ----------
    # 循环次数约为 S（=len(k_on_array)）
    for i in range(k_on_array.shape[0]):
        # 当前段 i 的 P(ON->ON)（ON 的“粘性”/持续概率）
        cur_kon = k_on_array[i]

        # 当前段的 2x2 转移矩阵 A^(i)
        # 行 0：上一帧是 OFF 时的分布；行 1：上一帧是 ON 时的分布。
        # 这里保持 P(OFF->OFF) 不变，只改变 P(ON->ON)。
        cur_transition_probabilities = np.zeros((2, 2))
        cur_transition_probabilities[0, 0] = k_off_off  # P(0->0)
        cur_transition_probabilities[0, 1] = 1 - k_off_off  # P(0->1)
        cur_transition_probabilities[1, 1] = cur_kon  # P(1->1) = 当前段的“ON粘性”
        cur_transition_probabilities[1, 0] = 1 - cur_kon  # P(1->0)

        # =========================================================
        # 生成“段 i”的序列：长度使用 (L + 1)
        # 目的：保留本段“最后一帧”作为下一段的“初值”，从而实现段与段的自然衔接。
        #       这样下一段不用再随机起步，避免段与段之间出现不自然的断点。
        # 返回形状：cur_chain_matrix.shape == (N, L+1)
        # 其中：
        #   - cur_chain_matrix[:, 0:L]   是本段需要写入到大矩阵的 L 帧内容（第一段）
        #   - cur_chain_matrix[:, 1:L+1] 是本段需要写入到大矩阵的 L 帧内容（后续段）
        #   - cur_chain_matrix[:, -1]    是“本段最后一帧”，作为下一段的 starting_value
        # =========================================================
        cur_chain_matrix = generate_promotor(
            cur_transition_probabilities,
            number_of_traces,
            segment_length + 1,
            starting_value
        )

        # 更新下一段的初值：= 本段最后一帧的状态（逐条轨迹对齐）
        # 形状：(N,)
        starting_value = cur_chain_matrix[:, -1]

        # =========================================================
        # 写回到大矩阵 chain_matrix：
        #   - 对第 0 段（i==0）：取本段的前 L 帧 [0:-1]，写入 [0:L]
        #   - 对第 1..S-1 段：取本段的后 L 帧 [1:]，写入 [i*L:(i+1)*L]
        #
        # 这样做的原因：
        #   - 我们用 (L+1) 的长度生成是为了“把最后一帧作为下一段初值”；
        #   - 如果我们把“最后一帧”也写回，那么下一段的“第 0 帧”将与上一段的“最后一帧”重复一次；
        #   - 因此需要在写回时丢弃一帧：
        #       * 第一段丢掉末尾那帧（[0:-1]）
        #       * 后续段丢掉开头那帧（[1:]）
        #   - 这样每段恰好贡献 L 帧，整个拼接后总长度 = S * L，且段与段在隐藏态上连续。
        # =========================================================
        if i == 0:
            chain_matrix[:, segment_save_index_start:segment_save_index_end] = cur_chain_matrix[:, 0:-1]
        else:
            chain_matrix[:, segment_save_index_start:segment_save_index_end] = cur_chain_matrix[:, 1:]

        # 推进写入区间到下一段的位置
        # 下一段写入 [segment_save_index_start : segment_save_index_end]
        segment_save_index_end += segment_length
        segment_save_index_start += segment_length

    promotor_csv_filename = (
        data_folder_head
        + "synthetic_make_data_promotor_dynamic_transition_segement_"
        + str(number_of_segement_for_dynamic_traces)
        + "_length_" + str(segment_length) + ".csv"
    )

# 保存 promoter 状态到 CSV（注意：CSV 会多出一列索引，后续读回需跳过）
sampling_dataframe = pd.DataFrame(chain_matrix)
print("The file is saved to: " + promotor_csv_filename)
sampling_dataframe.to_csv(promotor_csv_filename)

#%% 由 promoter 状态生成荧光轨迹（MS2 卷积 + 高斯噪声）

# 读回 CSV；skip_header=1 跳过表头；[:,1:] 去掉第一列索引
ms2_signals = genfromtxt(promotor_csv_filename, delimiter=',', skip_header=1)
signal_holder = ms2_signals[:, 1:]  # 形状：(n_traces, length_of_each_trace)
n_traces = len(signal_holder)
length_of_each_trace = len(signal_holder[0])
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
# 可选：计算序列开头（t < W）因核不完整的“尾部需减掉多少”（本脚本未用，调试/拓展时有用）
count_reduction_manual = np.zeros((1, W - 1))
for t in np.arange(0, W - 1):
    count_reduction_manual[0, t] = np.sum(ms2_coeff[0, t + 1:])
count_reduction_manual = np.reshape(count_reduction_manual, (W - 1, 1))

# ---- 用位运算维护“滑动窗口的复合状态编码” ----
# 以二进制编码最近 W 帧的 0/1（ON=1），如 W=2 时 mask=3（二进制 11）
mask = np.int32((2 ** W) - 1)

fluorescence_holder = np.zeros((n_traces, length_of_each_trace))
get_adjusted_trace_record = []  # 调试用：记录每帧 (ON权重, OFF权重)

for i in np.arange(0, len(fluorescence_holder)):
    tmp_get_adjusted_trace_record = []
    single_promoter = np.expand_dims(signal_holder[i, :], axis=0)  # (1, T) 的 0/1 序列
    single_trace = np.zeros((1, length_of_each_trace))

    # ----- 第 0 帧：窗口只有1位（历史为空），直接根据当前状态计算 -----
    t = 0
    window_storage = int(single_promoter[0, 0])  # 仅当前帧
    weights = get_adjusted(window_storage, K, W, ms2_coeff)  # 返回形如 [ON_weight, OFF_weight]
    # 期望均值 = ON_weight*mu_on + OFF_weight*mu_off；再加高斯噪声
    single_trace[0, t] = (weights[0] * mu[1, 0] + weights[1] * mu[0, 0]) + np.random.normal(0, noise)
    tmp_get_adjusted_trace_record.append(weights)

    # ----- 从第 1 帧起：利用位运算维护“最近 W 帧”的编码 -----
    window_storage = 0
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
        tmp_get_adjusted_trace_record.append(weights)

    fluorescence_holder[i, :] = single_trace
    get_adjusted_trace_record.append(tmp_get_adjusted_trace_record)

# 保存荧光到 CSV / MAT
sampling_dataframe = pd.DataFrame(fluorescence_holder)
if onoff_dynamic_transition:
    fluorescence_csv_name = (
        data_folder_head
        + "fluorescent_traces_synthetic_data_W_" + str(W)
        + "_trace_" + str(n_traces)
        + "_noise_" + str(noise)
        + "_dynamic_segement_" + str(number_of_segement_for_dynamic_traces)
        + "_length_" + str(segment_length) + ".csv"
    )
    fluorescence_mat_name = (
        data_folder_head
        + "fluorescent_traces_synthetic_data_W_" + str(W)
        + "_trace_" + str(n_traces)
        + "_noise_" + str(noise)
        + "_dynamic_segement_" + str(number_of_segement_for_dynamic_traces)
        + "_length_" + str(segment_length) + ".mat"
    )
else:
    fluorescence_csv_name = (
        data_folder_head
        + "code_check_fluorescent_traces_synthetic_data_W_" + str(W)
        + "_trace_" + str(n_traces)
        + "_noise_" + str(noise) + ".csv"
    )
    fluorescence_mat_name = (
        data_folder_head
        + "code_check_fluorescent_traces_synthetic_data_W_" + str(W)
        + "_trace_" + str(n_traces)
        + "_noise_" + str(noise) + ".mat"
    )

sampling_dataframe.to_csv(fluorescence_csv_name)

# ✅ 保存更多元数据：seed、真实的段参数等；避免硬编码 2/40 导致歧义
savemat(
    fluorescence_mat_name,
    {
        'Input': fluorescence_holder,                   # 注意命名：这里是“观测荧光”
        'Output': signal_holder,                        # 这里是“promoter 真值（0/1）”
        'deltaT': deltaT,
        'k_on_array': k_on_array,                       # 稳态时为标量；动态时为数组
        'k_off_off': k_off_off,
        'W': W,
        't_MS2': t_MS2,
        'noise': noise,
        'mean_on': mu[1, 0],
        'mean_off': mu[0, 0],
        'number_of_segement_for_dynamic_traces': number_of_segement_for_dynamic_traces,
        'segment_length': segment_length,
        'seed': int(seed_setter),                       # 新增：记录随机种子，便于复现
        'ms2_coeff': ms2_coeff,                         # 新增：把核也存起来，后续对齐更方便
        # 可选：'weights_record': np.array(get_adjusted_trace_record, dtype=object)  # 体积较大，按需打开
    }
)
print("finish code")

#%% 可视化：对比 promoter 真值(阶梯图) 与 合成荧光(曲线)
# 你也可以循环画多条，或随机抽样画若干条
plt.figure(15)
plt.step(synthetic_x, signal_holder[40, :], where='post')
plt.title('Promoter state (trace 40)')
plt.figure(16)
plt.plot(synthetic_x, fluorescence_holder[40, :].flatten())
plt.title('Fluorescence (trace 40)')

plt.figure(17)
plt.step(synthetic_x, signal_holder[42, :], where='post')
plt.title('Promoter state (trace 42)')
plt.figure(18)
plt.plot(synthetic_x, fluorescence_holder[42, :].flatten())
plt.title('Fluorescence (trace 42)')

plt.show()
