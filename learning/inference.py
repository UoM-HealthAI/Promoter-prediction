import matplotlib.pyplot as plt
import numpy as np
import torch
from model import FluorToBinaryCausal
from dataset import  ema_standardize
import evaluation

# Pick a random synthetic example
random_index = np.random.randint(0, 500000)
print(f"Random index picked: {random_index}")
synthetic_data_path = "../dataset/test/fluorescence_traces.npy"
prompter_data_path = "../dataset/test/promoter_states.npy"
synthetic_data = np.load(synthetic_data_path, allow_pickle=True)
prompter_data = np.load(prompter_data_path, allow_pickle=True)
fluorescence_trace = synthetic_data[random_index]
promoter_trace = prompter_data[random_index]
print(f"Fluorescence trace shape: {fluorescence_trace.shape}")
print(f"Promoter trace shape: {promoter_trace.shape}")
print(f"Promoter trace: {promoter_trace}")

# Plot the synthetic fluorescence trace and promoter states
plt.figure(figsize=(10, 5))
for idx in range(1):
    plt.subplot(1, 1, idx + 1)
    plt.plot(fluorescence_trace, label='Fluorescence Trace', color='blue')
    plt.step(range(len(promoter_trace)), promoter_trace * np.max(fluorescence_trace) * 0.6, where='post', label='Promoter State (scaled)', color='orange')
    plt.ylim([-1000, np.max(fluorescence_trace) * 1.1])
    plt.xlabel('Time (frames)')
    plt.ylabel('Fluorescence / Promoter State')
    plt.title(f'Synthetic Example Index {random_index}')
    plt.legend()
plt.tight_layout()
plt.show()

# Load the trained model and perform inference
model_path = "../logs/Promotor_States_Prediction/best_param.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)
parameters = torch.load(model_path, map_location=device)
# Remove 'module.' prefix if model was trained using DataParallel
parameters = {k.replace('module.', ''): v for k, v in parameters.items()}
print("parameters loaded")
model = FluorToBinaryCausal(d_model=256, nhead=8, num_layers=8, ff=512, dropout=0.1, max_len=200).to(device)
model.load_state_dict(parameters)
model.eval()
print("model loaded")

with torch.no_grad():
    fluorescence_trace_standardize = ema_standardize(fluorescence_trace, alpha=0.01, eps=1e-6, clip_val=6.0)
    fluorescence_trace_standardize = torch.tensor(fluorescence_trace_standardize, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T)
    logits = model(fluorescence_trace_standardize)  # (1, T)
    predictions = (torch.sigmoid(logits) > 0.5).long().squeeze(0).cpu().numpy()  # (T,)
    print(f"logits: {torch.sigmoid(logits)}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")

# plot the predictions against the ground truth
plt.figure(figsize=(10, 5))
for idx in range(1):
    plt.subplot(1, 1, idx + 1)
    plt.plot(fluorescence_trace, label='Fluorescence Trace', color='blue')
    plt.step(range(len(promoter_trace)), promoter_trace * np.max(fluorescence_trace) * 0.6, where='post', label='True Promoter State (scaled)', color='orange')
    plt.step(range(len(predictions)), predictions * np.max(fluorescence_trace) * 0.6, where='post', label='Predicted Promoter State (scaled)', color='green', linestyle='--')
    plt.ylim([-1000, np.max(fluorescence_trace) * 1.1])
    plt.xlabel('Time (frames)')
    plt.ylabel('Fluorescence / Promoter State')
    plt.title(f'Inference on Synthetic Example Index {random_index}')
    plt.legend()
plt.tight_layout()
plt.show()

# plot confusion matrix and evaluation metrics
cm = evaluation.confusion_matrix(promoter_trace, predictions)
print("Confusion Matrix:", cm)
evaluation.plt_confusion_matrix(cm)
evaluation.plot_diagrams(promoter_trace, predictions)
# ---------------------------------------------------
