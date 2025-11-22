import matplotlib.pyplot as plt
import numpy as np
import torch
from model import FluorToBinaryCausal
from dataset import  ema_standardize

# ---------------------------------------------------
# Load the fluorescence trace got from HMM inference in Hoppe et al. (2020)
fluorescence_trace_path = "../example_datasets/hoppe_et_al_ush_real_data_example/SINGLE_CELL_SIGNALS.csv"
# load the fluorescence trace
fluorescence_trace = np.genfromtxt(fluorescence_trace_path, delimiter=',', skip_header=1)
print("Fluorescence trace shape (including indices):", fluorescence_trace.shape)
# Randomly pick one trace
random_index = np.random.randint(0, fluorescence_trace.shape[0])
print(f"Random index picked: {random_index}")
fluorescence_trace = fluorescence_trace[random_index, 1:]  # remove the index column
print("Fluorescence trace shape (before trimming):", fluorescence_trace.shape)

promoter_trace_path = "../example_datasets/hoppe_et_al_ush_real_data_example/SINGLE_CELL_POSTERIOR.csv"
# load the predicted promoter states
promoter_trace = np.genfromtxt(promoter_trace_path, delimiter=',', skip_header=1)
print("Promoter trace shape (including indices):", promoter_trace.shape)
# Randomly pick one trace
promoter_trace = promoter_trace[random_index, 1:]  # remove the index column
print("Promoter trace shape (before trimming):", promoter_trace.shape)

# Match the fluorescence and promoter traces, if null, trim them
length_before_null = len(fluorescence_trace)
real_fluorescence_trace = fluorescence_trace.copy()
for i in range(length_before_null):
    if np.isnan(fluorescence_trace[i]) or np.isnan(promoter_trace[i]):
        real_fluorescence_trace = fluorescence_trace[:i]
        promoter_trace = promoter_trace[:i]
        break
print("Fluorescence trace shape (after trimming):", real_fluorescence_trace.shape)
print("Promoter trace shape (after trimming):", promoter_trace.shape)

# ---------------------------------------------------

# Plot the real fluorescence trace
plt.figure(figsize=(10, 5))
for idx in range(1):
    plt.subplot(1, 1, idx + 1)
    plt.plot(real_fluorescence_trace, label='Real Fluorescence Trace', color='blue')
    plt.ylim([np.min(real_fluorescence_trace), np.max(real_fluorescence_trace)])
    plt.xlabel('Time (frames)')
    plt.ylabel('Fluorescence')
    plt.title(f'Real Fluorescence Trace from Hoppe et al. (2020)')
    plt.legend()
plt.tight_layout()
plt.show()

# Load the trained model and perform inference
model_path = "../logs/Promotor_States_Prediction/best_param.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if
    torch.backends.mps.is_available() else "cpu")
print("device:", device)
parameters = torch.load(model_path, map_location=device)
# Remove the 'module.' prefix if the model was trained using DataParallel
parameters = {k.replace('module.', ''): v for k, v in parameters.items()}
print("parameters loaded")
model = FluorToBinaryCausal(d_model=256, nhead=8, num_layers=8, ff=512, dropout=0.1, max_len=200).to(device)
model.load_state_dict(parameters)
model.eval()
print("model loaded")

with torch.no_grad():
    real_fluorescence_trace_standardize = ema_standardize(real_fluorescence_trace, alpha=0.01, eps=1e-6, clip_val=6.0)
    real_fluorescence_trace_standardize = torch.tensor(real_fluorescence_trace_standardize, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T)
    logits = model(real_fluorescence_trace_standardize)  # (1, T)
    predictions = (torch.sigmoid(logits) > 0.5).long().squeeze(0).cpu().numpy()  # (T,)
    print(f"logits: {torch.sigmoid(logits)}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")

# plot the predictions
plt.figure(figsize=(10, 5))
for idx in range(1):
    plt.subplot(1, 1, idx + 1)
    plt.plot(real_fluorescence_trace, label='Real Fluorescence Trace', color='blue')
    plt.step(range(len(predictions)), predictions * np.max(real_fluorescence_trace) * 0.6, where='post', label='Predicted Promoter State (scaled)', color='orange')
    plt.ylim([np.min(real_fluorescence_trace), np.max(real_fluorescence_trace) * 1.1])
    plt.xlabel('Time (frames)')
    plt.ylabel('Fluorescence / Predicted Promoter State')
    plt.title(f'Predicted Promoter States from Real Fluorescence Trace')
    plt.legend()
plt.tight_layout()
plt.show()

# plot the predictions against the HMM predictions
plt.figure(figsize=(10, 5))
for idx in range(1):
    plt.subplot(1, 1, idx + 1)
    plt.plot(real_fluorescence_trace, label='Real Fluorescence Trace', color='blue')
    plt.step(range(len(promoter_trace)), promoter_trace * np.max(real_fluorescence_trace) * 0.6, where='post', label='HMM Predicted Promoter State (scaled)', color='orange')
    plt.step(range(len(predictions)), predictions * np.max(real_fluorescence_trace) * 0.6, where='post', label='DL Predicted Promoter State (scaled)', color='green', linestyle='--')
    plt.ylim([np.min(real_fluorescence_trace), np.max(real_fluorescence_trace) * 1.1])
    plt.xlabel('Time (frames)')
    plt.ylabel('Fluorescence / Promoter State')
    plt.title(f'Inference on Real Fluorescence Trace')
    plt.legend()
plt.tight_layout()
plt.show()
