import numpy as np
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
    """
    Compute the accuracy of predictions.

    Parameters:
    y_true (ndarray): Ground truth binary labels (0 or 1).
    y_pred (ndarray): Predicted binary labels (0 or 1).

    Returns:
    float: Accuracy score.
    """
    assert y_true.shape == y_pred.shape, "Shapes of true and predicted labels must match."
    correct = np.sum(y_true == y_pred)
    total = y_true.shape[0]
    return correct / total if total > 0 else 0.0

def precision(y_true, y_pred):
    """
    Compute the precision of predictions.

    Parameters:
    y_true (ndarray): Ground truth binary labels (0 or 1).
    y_pred (ndarray): Predicted binary labels (0 or 1).

    Returns:
    float: Precision score.
    """
    assert y_true.shape == y_pred.shape, "Shapes of true and predicted labels must match."
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def recall(y_true, y_pred):
    """
    Compute the recall of predictions.

    Parameters:
    y_true (ndarray): Ground truth binary labels (0 or 1).
    y_pred (ndarray): Predicted binary labels (0 or 1).

    Returns:
    float: Recall score.
    """
    assert y_true.shape == y_pred.shape, "Shapes of true and predicted labels must match."
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def f1_score(y_true, y_pred):
    """
    Compute the F1 score of predictions.

    Parameters:
    y_true (ndarray): Ground truth binary labels (0 or 1).
    y_pred (ndarray): Predicted binary labels (0 or 1).

    Returns:
    float: F1 score.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix of predictions.

    Parameters:
    y_true (ndarray): Ground truth binary labels (0 or 1).
    y_pred (ndarray): Predicted binary labels (0 or 1).

    Returns:
    dict: Confusion matrix with keys 'TP', 'TN', 'FP', 'FN'.
    """
    assert y_true.shape == y_pred.shape, "Shapes of true and predicted labels must match."
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

def plt_confusion_matrix(cm):
    """
    Plot the confusion matrix.

    Parameters:
    cm (dict): Confusion matrix with keys 'TP', 'TN', 'FP', 'FN'.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    labels = ['Positive', 'Negative']
    matrix = np.array([[cm['TP'], cm['FN']],
                       [cm['FP'], cm['TN']]])

    fig, ax = plt.subplots()
    im = ax.matshow(matrix, cmap=plt.cm.Blues)
    fig.colorbar(im)
    ax.set_title('Confusion Matrix')

    # ✅ 先固定刻度位置，再设置标签（长度要匹配）
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # 让 x 轴标签显示在底部（matshow 默认在顶部）
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # 在格子里标数字
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val}', ha='center', va='center',
                color='red')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.tight_layout()
    plt.show()

def plot_diagrams(y_true, y_pred):
    """
    Plot accuracy, precision, recall, and F1 score.

    Parameters:
    y_true (ndarray): Ground truth binary labels (0 or 1).
    y_pred (ndarray): Predicted binary labels (0 or 1).
    """
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics = [acc, prec, rec, f1]
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    plt.figure(figsize=(8, 5))
    plt.bar(labels, metrics, color=['blue', 'orange', 'green', 'red'])
    plt.ylim(0, 1)
    plt.title('Evaluation Metrics')
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Simulated ground truth and predictions
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 0])

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:", cm)

    plt_confusion_matrix(cm)
    plot_diagrams(y_true, y_pred)
