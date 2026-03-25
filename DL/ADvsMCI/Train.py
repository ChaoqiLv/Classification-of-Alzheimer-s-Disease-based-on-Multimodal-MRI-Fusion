import os
import sys
import socket
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, classification_report, matthews_corrcoef
import matplotlib.pyplot as plt
import nibabel as nib
import random
from model.ResNet import resnet18_3d, resnet10_3d, resnet34_3d, resnet50_3d, resnet101_3d, resnet152_3d, resnet200_3d
import json
import copy
import itertools
import time
from sklearn.metrics import balanced_accuracy_score
import shutil


# ================================== Save test set ROC data ==================================
def save_roc_data(save_dir, fold, phase, fpr, tpr, roc_auc):
    roc_dir = os.path.join(save_dir, 'ROC', phase)
    os.makedirs(roc_dir, exist_ok=True)
    uniform_fpr = np.linspace(0, 1, 100)
    uniform_tpr = np.interp(uniform_fpr, fpr, tpr)
    uniform_fpr[0], uniform_tpr[0] = 0, 0
    uniform_fpr[-1], uniform_tpr[-1] = 1, 1
    roc_df = pd.DataFrame({
        'FPR': uniform_fpr,
        'TPR': uniform_tpr,
        'AUC': [roc_auc] * len(uniform_fpr)
    })
    roc_df.to_csv(os.path.join(roc_dir, f'roc_data_{phase}_fold_{fold}.csv'), index=False)
    np.savez(os.path.join(roc_dir, f'roc_data_{phase}_fold_{fold}.npz'),
             fpr=uniform_fpr, tpr=uniform_tpr, roc_auc=roc_auc)
    plt.figure(figsize=(8, 6))
    plt.plot(uniform_fpr, uniform_tpr, color='blue', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({phase.capitalize()} Set, Fold {fold})')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(roc_dir, f'roc_curve_fold_{fold}.png'),
                dpi=1200, bbox_inches='tight')
    plt.close()


# ================================== Data Augmentation ==================================
class MultiModalDataset(Dataset):
    """Multi-modal 3D medical image dataset loader"""

    def __init__(self, features, labels, training=True):
        self.features = features
        self.labels = labels
        self.training = training

    def __len__(self):
        return len(self.labels) * (5 if self.training else 1)

    def __getitem__(self, idx):
        """Get single sample"""
        try:
            original_idx = idx // 5 if self.training else idx
            paths = self.features[original_idx]
            label = self.labels[original_idx]

            modalities = []
            for path in paths:
                if isinstance(path, np.ndarray):
                    path = path[0]

                # Read NIfTI file and convert to tensor
                img_data = nib.load(path).get_fdata()
                img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)
                img_tensor = self.resize_tensor(img_tensor)

                if self.training:  # Data augmentation
                    aug_index = idx % 5
                    if aug_index < 1:  # Translation augmentation
                        img_tensor = self.translate(img_tensor, aug_index)
                    elif aug_index < 2:  # Rotation augmentation
                        img_tensor = self.rotate(img_tensor, aug_index - 1)
                    elif aug_index < 3:  # Scaling augmentation
                        img_tensor = self.scale(img_tensor, aug_index - 2)
                    elif aug_index < 4:  # Noise augmentation
                        img_tensor = self.add_noise(img_tensor, aug_index - 3)
                    else:  # Brightness augmentation
                        img_tensor = self.adjust_brightness(img_tensor, aug_index - 4)

                modalities.append(img_tensor)

            return modalities, label

        except Exception as e:
            print(f"Error loading sample at index {idx}: {e}")
            return None, None

    # Data augmentation methods below
    def translate(self, img_tensor, direction):
        """Translation augmentation"""
        shift = (-3, -3, -3)
        if direction == 0:
            img_tensor = torch.roll(img_tensor, shifts=shift, dims=(1, 2, 3))
        return self.resize_tensor(img_tensor)

    def rotate(self, img_tensor, plane):
        """Rotation augmentation"""
        img_tensor = img_tensor.clone()
        if plane == 0:
            img_tensor = img_tensor.rot90(1, dims=(1, 2))
            img_tensor = img_tensor.rot90(1, dims=(2, 3))
            img_tensor = img_tensor.rot90(1, dims=(1, 3))
        return self.resize_tensor(img_tensor)

    def scale(self, img_tensor, factor):
        """Scaling augmentation"""
        scale_factors = [1.05]
        img_tensor_5d = img_tensor.unsqueeze(0)
        img_tensor_5d = nn.functional.interpolate(
            img_tensor_5d, scale_factor=scale_factors[factor],
            mode='trilinear', align_corners=False)
        return self.resize_tensor(img_tensor_5d.squeeze(0))

    def add_noise(self, img_tensor, noise_type):
        """Add noise"""
        noise_levels = [0.05]
        noise = torch.randn_like(img_tensor) * noise_levels[noise_type]
        return img_tensor + noise

    def adjust_brightness(self, img_tensor, factor):
        """Brightness augmentation"""
        brightness_factor = [1.05]
        img_tensor = img_tensor.clone()
        img_tensor = img_tensor * brightness_factor[factor]
        return img_tensor

    def resize_tensor(self, img_tensor):
        """Resize tensor to fixed size (1,61,73,61)"""
        desired_shape = (1, 61, 73, 61)
        current_shape = img_tensor.shape[1:]
        resized_tensor = img_tensor.clone()

        # Depth adjustment
        if current_shape[0] > desired_shape[1]:
            resized_tensor = resized_tensor[:, :desired_shape[1], :, :]
        else:
            pad_depth = (desired_shape[1] - current_shape[0]) // 2
            resized_tensor = nn.functional.pad(resized_tensor, (0, 0, 0, 0, pad_depth, pad_depth))

        # Height adjustment
        if current_shape[1] > desired_shape[2]:
            resized_tensor = resized_tensor[:, :, :desired_shape[2], :]
        else:
            pad_height = (desired_shape[2] - current_shape[1]) // 2
            resized_tensor = nn.functional.pad(resized_tensor, (0, 0, pad_height, pad_height, 0, 0))

        # Width adjustment
        if current_shape[2] > desired_shape[3]:
            resized_tensor = resized_tensor[:, :, :, :desired_shape[3]]
        else:
            pad_width = (desired_shape[3] - current_shape[2]) // 2
            resized_tensor = nn.functional.pad(resized_tensor, (pad_width, pad_width, 0, 0, 0, 0))

        return self.fill_zeros_to_shape(resized_tensor, desired_shape)

    def fill_zeros_to_shape(self, img_tensor, desired_shape):
        """Fill with zeros to ensure target shape"""
        _, depth, height, width = img_tensor.shape
        _, D, H, W = desired_shape
        out = img_tensor.clone()

        # Depth padding/cropping
        if depth < D:
            out = nn.functional.pad(out, (0, 0, 0, 0, 0, D - depth))
        elif depth > D:
            out = out[:, :D, :, :]

        # Height padding/cropping
        if out.shape[2] < H:
            out = nn.functional.pad(out, (0, 0, 0, H - out.shape[2], 0, 0))
        elif out.shape[2] > H:
            out = out[:, :, :H, :]

        # Width padding/cropping
        if out.shape[3] < W:
            out = nn.functional.pad(out, (0, W - out.shape[3], 0, 0, 0, 0))
        elif out.shape[3] > W:
            out = out[:, :, :, :W]

        return out


# ================================= Logger =================================
class Logger(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ================================= Early Stopping Strategy =================================
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0.0, save_dir='model_result', fold=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss_score = None
        self.best_acc_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = 0.0
        self.delta = delta
        self.save_dir = save_dir
        self.fold = fold
        self.best_epoch = 0
        self.best_modality_weights = None
        self.best_model_states = None
        self.best_roc_data = {}

    def __call__(self, val_loss, val_acc, models, modality_weights, epoch):
        # Initialize on first call and save model
        if self.best_acc_score is None:
            print("Initial training:")
            if self.verbose:
                print(f"Accuracy improved ({self.val_acc_max:.4f} → {val_acc:.4f}), saving model...")
            self.best_loss_score = val_loss
            self.best_acc_score = val_acc
            self._save_checkpoint(val_loss, val_acc, models, modality_weights, epoch)
            return

        # Determine current model performance
        if val_acc < self.best_acc_score:
            # Accuracy decreased, increment early stopping counter
            self.counter += 1
            if self.verbose:
                print(f"Current accuracy {val_acc:.4f} < best accuracy {self.best_acc_score:.4f}")
                print(f'Early stopping counter: {self.counter}/{self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered, best model from epoch {self.best_epoch + 1}")
                print("=" * 50)

        elif val_acc > self.best_acc_score + self.delta:
            # Accuracy improved, save model and reset counter
            if self.verbose:
                print(f"\nAccuracy improved ({self.val_acc_max:.4f} → {val_acc:.4f}), saving model...")
            self.best_acc_score = val_acc
            self.best_loss_score = val_loss
            self._save_checkpoint(val_loss, val_acc, models, modality_weights, epoch)
            self.counter = 0

        elif val_acc == self.best_acc_score and val_loss < self.best_loss_score + self.delta:
            # Same accuracy but lower loss, save model and reset counter
            if self.verbose:
                print(f"\nSame accuracy but loss decreased ({self.val_loss_min:.4f} → {val_loss:.4f}), saving model...")
            self.best_loss_score = val_loss
            self._save_checkpoint(val_loss, val_acc, models, modality_weights, epoch)
            self.counter = 0

        else:
            self.counter += 1
            if self.verbose:
                print(f'Early stopping counter: {self.counter}/{self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered, best model from epoch {self.best_epoch + 1}")
                print("=" * 50)

    def _save_checkpoint(self, val_loss, val_acc, models, modality_weights, epoch):
        """Save best model and weights"""
        self.best_epoch = epoch
        self.val_acc_max = val_acc
        self.val_loss_min = val_loss
        self.best_modality_weights = modality_weights.clone()
        self.best_model_states = [copy.deepcopy(model.state_dict()) for model in models]

        # Save model weights
        for i, state_dict in enumerate(self.best_model_states):
            torch.save(state_dict,
                       os.path.join(self.save_dir, 'ResNet',
                                    f'ResNet_{self.fold + 1}_model_{i}.pth'))

        # Save modality weights
        weights_dict = {
            'modality_weights': self.best_modality_weights.cpu().numpy().tolist(),
            'best_epoch': epoch + 1,
            'best_val_acc': val_acc.item(),
            'best_val_loss': val_loss,
        }
        with open(os.path.join(self.save_dir, f'ResNet_{self.fold + 1}_modality_weights.json'), 'w') as f:
            json.dump(weights_dict, f, indent=4)

        # Print model parameters (first 5 weights of classification layer)
        print("\n=== Best Model Parameters Validation ===")
        for i, model in enumerate(models):
            print(f"Model {i} last layer weights:")
            print(model.fc.weight.data[0, :5])

        # Print modality weights
        print("\n=== Best Modality Weights ===")
        print(self.best_modality_weights.cpu().numpy().tolist())


# ================================= Plot Confusion Matrix =================================
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, normalize=False):
    plt.figure(figsize=(8, 6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        if np.issubdtype(cm.dtype, np.integer):
            fmt = 'd'
        else:
            fmt = '.2f'
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# ================================= Training Code =================================
def train_model(fold, num_epochs, lr, device, save_dir, train_dataloader, val_dataloader, test_dataloader):
    # ================================= Model Initialization =================================
    models = [resnet18_3d().to(device) for _ in range(4)]

    # Define modality weights
    modality_weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)  # GM, FA, MD, IC

    # ================================= Load Pretrained Weights =================================
    pretrained_path = "resnet_18.pth"
    checkpoint = torch.load(pretrained_path, map_location=device)
    pretrained_dict = checkpoint.get('state_dict', checkpoint)

    # Remove 'module.' prefix
    new_pretrained_dict = {k[7:] if k.startswith('module.') else k: v for k, v in pretrained_dict.items()}

    for i, model in enumerate(models):
        model_dict = model.state_dict()
        # Only load weights with matching shapes
        filtered_dict = {k: v for k, v in new_pretrained_dict.items()
                         if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"Model {i} loaded pretrained weights from {len(filtered_dict)} layers")

    # ================================= Loss Function and Optimizer =================================
    criterion = nn.CrossEntropyLoss()

    # optimizers = [optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4) for model in models]
    # optimizers = [optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) for model in models]
    # optimizers = [optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) for model in models]
    optimizers = [optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-4) for model in models]

    schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99) for optimizer in optimizers]

    # ================================= Logging and Recording =================================
    log_dir = os.path.join(save_dir, 'ResNet', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)
    print(trainval_sizes, "Test:", test_size)

    # ================================= Initialize Metrics =================================
    metrics = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'test_accuracy': [], 'test_precision': [],
        'test_recall': [], 'test_f1': [],
        'test_bacc': [], 'train_bacc': [],
        'roc_data': [],
        'train_mcc': [], 'val_mcc': []
    }

    # ================================= Initialize Early Stopping Instance =================================
    early_stopping = EarlyStopping(patience=20, verbose=True, delta=0.00001, save_dir=save_dir, fold=fold)

    # ================================= Training Loop =================================
    for epoch in range(num_epochs):
        print("*" * 75)
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0.0
            total_samples = 0
            all_preds = []
            all_labels = []

            # Set model mode
            for model in models:
                model.train() if phase == 'train' else model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase], desc=f'Epoch {epoch + 1}/{num_epochs} {phase}', disable=True):
                inputs = [input_.to(device) for input_ in inputs]
                labels = labels.long().to(device)
                batch_size = labels.size(0)
                total_samples += batch_size

                # Clear gradients
                if phase == 'train':
                    for optimizer in optimizers:
                        optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    # Compute output for each modality independently
                    all_logits = [model(input_) for model, input_ in zip(models, inputs)]
                    # Compute output probabilities for each modality
                    all_probs = [torch.softmax(logits, dim=1) for logits in all_logits]
                    # Perform weighted summation
                    weighted_probs = torch.stack([probs * w for probs, w in zip(all_probs, modality_weights)])
                    final_probs = torch.sum(weighted_probs, dim=0)

                    # Compute loss for each modality
                    modal_losses = [criterion(logits, labels) for logits in all_logits]
                    # Weighted sum using current modality weights
                    total_loss = torch.sum(torch.stack(modal_losses) * modality_weights)
                    # Multi-modal loss
                    loss = total_loss

                    # Predictions
                    _, preds = torch.max(final_probs, 1)

                    # Collect predictions and true labels
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    # Backward propagation
                    if phase == 'train':
                        total_loss.backward()
                        for optimizer in optimizers:
                            optimizer.step()

                # Statistics
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)

            # Update learning rate
            if phase == 'train':
                for scheduler in schedulers:
                    scheduler.step()

            # Calculate epoch metrics
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            epoch_mcc = matthews_corrcoef(all_labels, all_preds)

            print(f"[{phase}] Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} MCC: {epoch_mcc:.4f}")

            # Record metrics
            metrics[f'{phase}_loss'].append(epoch_loss)
            metrics[f'{phase}_accuracy'].append(epoch_acc.item())
            metrics[f'{phase}_mcc'].append(epoch_mcc)
            writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/accuracy', epoch_acc, epoch)
            writer.add_scalar(f'{phase}/mcc', epoch_mcc, epoch)

            # Automatic modality weight update
            if phase == 'val':
                # Temporarily store predictions and true labels for each modality
                all_modal_preds = [[] for _ in range(4)]
                all_labels = []

                for inputs, labels in val_dataloader:
                    inputs = [input_.to(device) for input_ in inputs]
                    labels = labels.long().to(device)

                    with torch.no_grad():
                        # Get independent outputs for each modality
                        modal_logits = [model(input_) for model, input_ in zip(models, inputs)]
                        modal_probs = [torch.softmax(logits, dim=1) for logits in modal_logits]

                        # Record predictions for each modality
                        for i in range(4):
                            _, preds = torch.max(modal_probs[i], 1)
                            all_modal_preds[i].extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                # Calculate validation accuracy for each modality
                modal_accuracies = []
                for i in range(4):
                    acc = np.mean(np.array(all_modal_preds[i]) == np.array(all_labels))
                    modal_accuracies.append(acc)

                # Update weights based on accuracy (Softmax normalization)
                acc_tensor = torch.tensor(modal_accuracies, device=device)
                new_weights = torch.softmax(acc_tensor / 0.1, dim=0)        # Temperature coefficient 0.1 controls weight differences
                new_weights = torch.clamp(new_weights, min=0.05)            # Each modality at least 5% weight
                new_weights = new_weights / new_weights.sum()               # Re-normalize

                # Smooth update (avoid sudden weight changes)
                modality_weights = 0.5 * modality_weights + 0.5 * new_weights

                print(f"Modality weights update: {modality_weights.cpu().numpy().round(4)}")
                print(f"Modality accuracies: {np.array(modal_accuracies).round(4)}")

            # Call early stopping after validation phase
            if phase == 'val':
                early_stopping(epoch_loss, epoch_acc, models, modality_weights, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch: {epoch + 1}")
            break

    # ================== Plot Training Curves ==================
    # Ensure save directory exists
    os.makedirs(os.path.join(save_dir, 'statistics'), exist_ok=True)

    # Create subdirectory for current fold
    fold_stats_dir = os.path.join(save_dir, 'statistics', f'fold_{fold + 1}')
    os.makedirs(fold_stats_dir, exist_ok=True)

    # Save training results to CSV
    train_results = pd.DataFrame({
        'Epoch': range(1, len(metrics['train_loss']) + 1),
        'loss': metrics['train_loss'],
        'val_loss': metrics['val_loss'],
        'train_acc': metrics['train_accuracy'],
        'val_acc': metrics['val_accuracy'],
        'train_mcc': metrics['train_mcc'],
        'val_mcc': metrics['val_mcc']
    })
    train_results.to_csv(os.path.join(fold_stats_dir, 'train_results.csv'), index=False)

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.title(f"Loss During Training and Validating (Fold {fold + 1})")
    x = np.arange(1, len(metrics['train_loss']) + 1)
    plt.plot(x, metrics['train_loss'], label="Train")
    plt.plot(x, metrics['val_loss'], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fold_stats_dir, 'train_epoch_losses.png'), dpi=1200, bbox_inches='tight')
    plt.close()

    # Plot accuracy curve
    plt.figure(figsize=(10, 6))
    plt.title(f"Accuracy During Training and Validating (Fold {fold + 1})")
    x = np.arange(1, len(metrics['train_accuracy']) + 1)
    plt.plot(x, metrics['train_accuracy'], label="Train")
    plt.plot(x, metrics['val_accuracy'], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fold_stats_dir, 'train_epoch_accuracy.png'), dpi=1200, bbox_inches='tight')
    plt.close()

    # Plot MCC curve
    plt.figure(figsize=(10, 6))
    plt.title(f"MCC During Training and Validating (Fold {fold + 1})")
    x = np.arange(1, len(metrics['train_mcc']) + 1)
    plt.plot(x, metrics['train_mcc'], label="Train")
    plt.plot(x, metrics['val_mcc'], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("MCC")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fold_stats_dir, 'train_epoch_mcc.png'), dpi=1200, bbox_inches='tight')
    plt.close()

    # Save summary data for all folds
    all_folds_dir = os.path.join(save_dir, 'statistics', 'all_folds')
    os.makedirs(all_folds_dir, exist_ok=True)
    train_results['Fold'] = fold + 1
    train_results.to_csv(os.path.join(all_folds_dir, f'train_results_fold_{fold + 1}.csv'), index=False)

    # ================== Test Evaluation ==================
    print(f"Testing with best validation model (epoch {early_stopping.best_epoch + 1}, acc {early_stopping.val_acc_max:.4f})...")

    # Initialize model architecture
    models = [resnet18_3d().to(device) for _ in range(4)]

    # Load best model from disk
    modality_weights = early_stopping.best_modality_weights
    best_epoch = early_stopping.best_epoch

    # Load best model and verify parameters
    print(f"\n=== Internal Test Set Model Parameter Validation: {best_epoch + 1} ===")
    for i, model in enumerate(models):
        model_path = os.path.join(save_dir, 'ResNet', f'ResNet_{fold + 1}_model_{i}.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist")
        # Load model parameters
        loaded_state_dict = torch.load(model_path)
        model.load_state_dict(loaded_state_dict)
        print(f"Model {i} loaded from {model_path}")

        model.eval()

        print(f"Model {i} last layer weights:")
        print(model.fc.weight.data[0, :5])

        # Count loaded layers and parameters
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in loaded_state_dict.items()
                         if k in model_dict and v.size() == model_dict[k].size()}
        print(f"Model {i} successfully loaded layers: {len(filtered_dict)}/{len(model_dict)}")
        print(f"Model {i} unloaded layers: {len(model_dict) - len(filtered_dict)}")
        print("-" * 50)

    # Print best modality weights
    print("\n=== Internal Test Set Best Modality Weights ===")
    print(modality_weights.cpu().numpy().tolist())
    print("=" * 50)
    print("\n=== Internal Test Set Best Epoch ===")

    all_test_probs = []
    all_test_preds = []
    all_test_labels = []

    for inputs, labels in tqdm(test_dataloader, desc='Testing', disable=True):
        inputs = [input_.to(device) for input_ in inputs]
        labels = labels.long().to(device)

        with torch.no_grad():
            # Get independent outputs for each modality
            modal_logits = [model(input_) for model, input_ in zip(models, inputs)]
            # Output softmax probabilities
            modal_probs = [torch.softmax(logits, dim=1) for logits in modal_logits]
            # Weighted summation based on weights
            weighted_probs = torch.stack([probs * w for probs, w in zip(modal_probs, modality_weights)])
            final_probs = torch.sum(weighted_probs, dim=0)

            # Predictions
            _, preds = torch.max(final_probs, 1)

            # Collect results
            all_test_probs.extend(final_probs.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    # Calculate and save test set ROC curve
    test_fpr, test_tpr, _ = roc_curve(all_test_labels, np.array(all_test_probs)[:, 1])
    test_roc_auc = roc_auc_score(all_test_labels, np.array(all_test_probs)[:, 1])
    save_roc_data(save_dir, fold + 1, 'test', test_fpr, test_tpr, test_roc_auc)

    # Calculate test metrics
    cm = confusion_matrix(all_test_labels, all_test_preds, labels=[0, 1])
    report = classification_report(all_test_labels, all_test_preds, digits=4)
    test_acc = np.mean(np.array(all_test_preds) == np.array(all_test_labels))
    precision = precision_score(all_test_labels, all_test_preds, average='macro')
    recall = recall_score(all_test_labels, all_test_preds, average='macro')
    f1 = f1_score(all_test_labels, all_test_preds, average='macro')
    auroc = roc_auc_score(all_test_labels, np.array(all_test_probs)[:, 1])
    mcc = matthews_corrcoef(all_test_labels, all_test_preds)
    bacc = balanced_accuracy_score(all_test_labels, all_test_preds)

    print("Test set metrics (macro average):")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Balanced Accuracy (BACC): {bacc:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Confusion Matrix: \n{cm}")
    print("====== Detailed Classification Report =====")
    print(report)

    # Return results
    return {
        'test_acc': test_acc,
        'bacc': bacc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'mcc': mcc,
        'cm': cm,
        'report': report,
        'best_epoch': early_stopping.best_epoch,
        'early_stopping': early_stopping,
        'all_test_probs': all_test_probs
    }


if __name__ == "__main__":
    # Fix random seed for reproducibility
    seed = 42
    print('seed is ' + str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set logging
    sys.stdout = Logger("training_log.txt")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============== Device being used by the model: =============")
    print(device)

    # Load data
    data_csv = pd.read_csv('data/data_HPPH_ADNI3.csv')
    class_pair = ('MCI', 'AD')
    modality_columns = ['Path_GM', 'Path_FA', 'Path_MD', 'Path_IC']

    # Data preprocessing
    data_subset = data_csv[data_csv['Label'].isin(class_pair)].copy()
    if data_subset.empty:
        raise ValueError(f"No data found for classes {class_pair} in CSV. Please check labels and class names.")

    # Label mapping: first class maps to 0, second class maps to 1
    label_mapping = {class_pair[0]: 0, class_pair[1]: 1}
    data_subset['Label'] = data_subset['Label'].map(label_mapping)

    # Check if all modality columns exist
    for col in modality_columns:
        if col not in data_subset.columns:
            raise ValueError(f"Specified modality column '{col}' does not exist in CSV.")

    X = data_subset[modality_columns].values
    y = data_subset['Label'].values
    print("============== Training Data Statistics =============")
    counts = pd.Series(y).value_counts().sort_index()
    for label_num, count in counts.items():
        label_name = {v: k for k, v in label_mapping.items()}[label_num]
        print(f"{label_name} ({label_num}) sample count: {count}")

    # Training parameters
    num_epochs = 100

    lr = 1e-3
    batch_size = 4

    save_dir = 'model_result'

    since = time.time()

    all_metrics = {
        'acc': [], 'precision': [], 'recall': [],
        'f1': [], 'auroc': [], 'mcc': [],
        'cm': [], 'report': [], 'bacc': []
    }

    # Five-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n========== Fold {fold + 1}/5 ==========")

        # Data split
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]

        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=seed)

        # Class balancing for training set
        train_df = pd.DataFrame({'X': list(X_train), 'y': y_train})
        print("Training set class distribution before downsampling:")
        print(train_df['y'].value_counts())

        min_count = train_df['y'].value_counts().min()
        balanced_df = pd.concat([
            train_df[train_df['y'] == label].sample(min_count, random_state=seed)
            for label in train_df['y'].unique()
        ]).sample(frac=1, random_state=seed)

        X_train_balanced = np.array(balanced_df['X'].tolist())
        y_train_balanced = balanced_df['y'].values
        print("Training set class distribution after downsampling:")
        print(pd.Series(y_train_balanced).value_counts())

        # Class balancing for validation set
        val_df = pd.DataFrame({'X': list(X_val), 'y': y_val})
        print("Validation set class distribution before downsampling:")
        print(val_df['y'].value_counts())

        val_min_count = val_df['y'].value_counts().min()
        val_balanced_df = pd.concat([
            val_df[val_df['y'] == label].sample(val_min_count, random_state=seed)
            for label in val_df['y'].unique()
        ]).sample(frac=1, random_state=seed)

        X_val_balanced = np.array(val_balanced_df['X'].tolist())
        y_val_balanced = val_balanced_df['y'].values
        print("Validation set class distribution after downsampling:")
        print(pd.Series(y_val_balanced).value_counts())

        # Create datasets
        train_dataset = MultiModalDataset(X_train_balanced, y_train_balanced, training=True)
        val_dataset = MultiModalDataset(X_val_balanced, y_val_balanced, training=False)
        test_dataset = MultiModalDataset(X_test, y_test, training=False)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Train and evaluate
        results = train_model(
            fold, num_epochs, lr, device, save_dir, train_loader, val_loader, test_loader)

        # Extract metrics from results
        all_metrics['acc'].append(results['test_acc'])
        all_metrics['bacc'].append(results['bacc'])
        all_metrics['precision'].append(results['precision'])
        all_metrics['recall'].append(results['recall'])
        all_metrics['f1'].append(results['f1'])
        all_metrics['auroc'].append(results['auroc'])
        all_metrics['mcc'].append(results['mcc'])
        all_metrics['cm'].append(results['cm'])
        all_metrics['report'].append(results['report'])

    # Internal test set results
    print("\n===== Test Set Five-Fold Cross Validation Results =====")
    print(f"Average Accuracy: {np.mean(all_metrics['acc']):.4f} ± {np.std(all_metrics['acc']):.4f}")
    print(f"Average Balanced Accuracy: {np.mean(all_metrics['bacc']):.4f} ± {np.std(all_metrics['bacc']):.4f}")
    print(f"Macro Average Precision: {np.mean(all_metrics['precision']):.4f} ± {np.std(all_metrics['precision']):.4f}")
    print(f"Macro Average Recall: {np.mean(all_metrics['recall']):.4f} ± {np.std(all_metrics['recall']):.4f}")
    print(f"Macro Average F1 Score: {np.mean(all_metrics['f1']):.4f} ± {np.std(all_metrics['f1']):.4f}")
    print(f"Average AUROC: {np.mean(all_metrics['auroc']):.4f} ± {np.std(all_metrics['auroc']):.4f}")
    print(f"Average MCC: {np.mean(all_metrics['mcc']):.4f} ± {np.std(all_metrics['mcc']):.4f}")
    print("\nConfusion Matrices:")
    for i, cm in enumerate(all_metrics['cm']):
        print(f"Fold {i + 1}:\n{cm}")

    # Calculate internal test set average confusion matrix
    avg_cm = np.mean([cm for cm in all_metrics['cm']], axis=0)
    class_names = [class_pair[0], class_pair[1]]

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(avg_cm, classes=class_names,
                          title='Average Confusion Matrix (Internal Test)')
    plt.savefig(os.path.join(save_dir, 'average_internal_confusion_matrix.png'), dpi=1200, bbox_inches='tight')
    plt.close()

    # Plot normalized confusion matrix
    plot_confusion_matrix(avg_cm, classes=class_names,
                          title='Normalized Average Confusion Matrix (Internal Test)',
                          normalize=True)
    plt.savefig(os.path.join(save_dir, 'average_internal_confusion_matrix_normalized.png'), dpi=1200, bbox_inches='tight')
    plt.close()

    sys.stdout.log.close()
    sys.stdout = sys.__stdout__
