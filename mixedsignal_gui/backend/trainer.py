from PySide6.QtCore import QThread, Signal
import numpy as np
import os
import time
import json
import traceback
import concurrent.futures
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import TensorDataset, DataLoader

from .torch_models import get_model, inspect_external_model

# Models that expect 2-channel IQ input
IQ_MODELS = {'ResNet1DOptimized'}


def pack_multichannel(a, target_len: int) -> np.ndarray:
    """Lay a multi-antenna capture out as model input channels.

    ``a`` is ``(n_ant, L)`` as produced by the antenna-array option in the
    stochastic and ray-tracing augmentations, which returns complex64.

    Complex input becomes ``2 * n_ant`` real channels ordered
    ``[Re(a0), Im(a0), Re(a1), Im(a1), ...]``; real input stays ``n_ant``
    channels.  For a single antenna the complex case is exactly ``[I, Q]``,
    the same layout the 1-D path produces, so the two agree by construction.

    This lives here, and is imported by both tabs, because the layout used to
    be re-derived in three places and they disagreed: the trainer cast the
    array to float32 (discarding Q outright) while the tabs flattened it, kept
    only antenna 0, and split *its* real and imaginary parts into the two
    channels.  A 2-antenna model trained at 100% then scored 50% at inference
    with no error, because the channel count still matched.
    """
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    n_ant = a.shape[0]
    L = min(a.shape[-1], target_len)
    if np.iscomplexobj(a):
        out = np.zeros((2 * n_ant, target_len), dtype=np.float32)
        out[0::2, :L] = np.real(a[:, :L])
        out[1::2, :L] = np.imag(a[:, :L])
    else:
        out = np.zeros((n_ant, target_len), dtype=np.float32)
        out[:, :L] = a[:, :L]
    return out


class TrainerThread(QThread):
    # Signals: epoch, total_epochs, train_loss, val_loss, train_acc, val_acc
    progress = Signal(int, int, float, float, float, float)
    finished = Signal(str)

    # Match the notebook's signal length for ResNet1D
    TARGET_LENGTH = 2048

    def __init__(self, file_label_pairs, labels, model_name='SimpleCNN', epochs=10,
                 batch_size=32, lr=0.001, val_split=0.2,
                 weight_decay=1e-4, label_smoothing=0.1, grad_clip=1.0,
                 model_hparams=None, model_plugin_path=None, save_dir=None, device='cpu'):
        super().__init__()
        self.file_label_pairs = list(file_label_pairs)
        self.labels = list(labels)
        self.model_name = model_name
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.val_split = float(val_split)
        self.weight_decay = float(weight_decay)
        self.label_smoothing = float(label_smoothing)
        self.grad_clip = float(grad_clip)
        self.model_hparams = model_hparams or {}
        self.model_plugin_path = model_plugin_path
        self.model_plugin = inspect_external_model(model_plugin_path) if model_plugin_path else None
        self.save_dir = save_dir
        
        # Use provided device parameter instead of auto-detecting
        self.device = torch.device(device)
        self._stop = False
        # Why the run produced no model, for the UI to show.  None on success.
        self.error = None
        
        if self.device.type == 'cuda':
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                print(f"Using device: {self.device} - {torch.cuda.get_device_name(0)}")
            else:
                print(f"Warning: CUDA device requested but not available, falling back to CPU")
                self.device = torch.device('cpu')
                print(f"Using device: {self.device}")
        else:
            print(f"Using device: {self.device}")

    def stop(self):
        """Request training to stop after the current epoch."""
        self._stop = True

    def _load_array(self, path):
        try:
            if path.lower().endswith('.npy') or path.lower().endswith('.npz'):
                arr = np.load(path, allow_pickle=True)
                # npz -> first array
                if isinstance(arr, np.lib.npyio.NpzFile):
                    keys = list(arr.keys())
                    arr = arr[keys[0]] if keys else None
                # Preserve complex baseband: forcing float32 here discarded Q
                # at load time, so no downstream code could ever see it.
                arr = np.asarray(arr)
                return arr if np.iscomplexobj(arr) else arr.astype(np.float32)
            if path.lower().endswith('.csv'):
                return np.loadtxt(path, delimiter=',').astype(np.float32)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
        return None

    @property
    def _is_iq_model(self):
        """Return True if the selected model expects 2-channel IQ input."""
        return self.model_name in IQ_MODELS

    def _prepare_iq_data(self, X_flat):
        """Split a complex signal into 2-channel (I, Q) format.

        Each row of *X_flat* is one complex baseband signal; the result is
        (N, 2, L) with I and Q as separate channels and L unchanged.

        Only complex input is accepted.  This previously fell back to treating
        a real array as interleaved [I0, Q0, I1, Q1, ...], which is wrong for
        the passband signals this app produces by default — those are real RF
        waveforms I*cos(wt) - Q*sin(wt), whose consecutive samples are two
        decimated copies of the same carrier rather than I and Q.  Real data is
        now routed to the single-channel path instead.
        """
        if not np.iscomplexobj(X_flat):
            raise ValueError(
                "_prepare_iq_data expects complex baseband; real signals should "
                "use _prepare_1ch_data.")
        return np.stack([X_flat.real, X_flat.imag], axis=1).astype(np.float32)

    def _prepare_1ch_data(self, X_flat):
        """Reshape flat data to (N, 1, L) for 1-channel Conv1d models."""
        return X_flat[:, np.newaxis, :]

    def _normalize_iq(self, X_iq):
        """Per-sample power normalization for IQ data (N, 2, L)."""
        power = np.mean(X_iq[:, 0, :] ** 2 + X_iq[:, 1, :] ** 2, axis=1, keepdims=True)
        power = np.maximum(power, 1e-10)
        scale = np.sqrt(power)[:, np.newaxis, :]  # (N, 1, 1)
        return X_iq / scale

    def run(self):
        if not self.file_label_pairs:
            self.finished.emit("")
            return

        # Load all samples
        # Load all samples concurrently
        print(f"Loading {len(self.file_label_pairs)} samples...")
        start_load = time.time()
        
        X_list = [None] * len(self.file_label_pairs)
        y_list = [None] * len(self.file_label_pairs)
        
        def load_one(idx, path, label):
            arr = self._load_array(path)
            if arr is not None:
                arr = np.asarray(arr)
                # Multi-channel arrays (num_rx_ant, N) are kept as-is
                if arr.ndim == 2 and arr.shape[0] <= 64:
                    return idx, arr, label
                return idx, arr.ravel(), label
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(load_one, i, p, l) for i, (p, l) in enumerate(self.file_label_pairs)]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    i, x, y_val = res
                    X_list[i] = x
                    y_list[i] = y_val
        
        # Filter Nones — keep X and y paired so indices stay aligned
        paired = [(x, y) for x, y in zip(X_list, y_list) if x is not None]
        X_list = [p[0] for p in paired]
        y_list = [p[1] for p in paired]
        
        print(f"Loaded {len(X_list)} samples in {time.time() - start_load:.2f}s")

        if not X_list:
            self.finished.emit("")
            return

        # Detect multi-channel arrays (2D with small first dim)
        is_multi_channel = any(
            isinstance(a, np.ndarray) and a.ndim == 2 for a in X_list
        )

        # Use fixed target length to match notebook preprocessing (2048)
        # Raw waveform files can be very long (e.g. 98304); passing them
        # untruncated makes convolutions ~24x slower than intended.
        target_len = self.TARGET_LENGTH
        if target_len <= 0:
            if is_multi_channel:
                target_len = max([a.shape[-1] for a in X_list])
            else:
                target_len = max([a.size for a in X_list])
        if target_len == 0:
            self.finished.emit("")
            return

        if is_multi_channel:
            # Multi-channel: stack as (batch, num_channels, target_len).
            # pack_multichannel keeps quadrature — this used to cast straight
            # to float32, discarding Q with only a ComplexWarning.
            packed = [pack_multichannel(a, target_len) for a in X_list]
            channel_counts = {p.shape[0] for p in packed}
            if len(channel_counts) > 1:
                raise ValueError(
                    f"Mixed channel counts in one training set: {sorted(channel_counts)}. "
                    f"Every file must have the same number of antennas, and be "
                    f"consistently real or complex.")
            X = np.stack(packed)
            input_channels = X.shape[1]
            was_complex = any(np.iscomplexobj(a) for a in X_list)
            note = (f" ({input_channels // 2} antenna(s) x I/Q)"
                    if was_complex else "")
            print(f"Multi-channel mode: {input_channels} channels{note}, "
                  f"{target_len} samples")
        else:
            # Pad/truncate to target length.  The buffer must stay complex when
            # the data is: casting to float32 here silently dropped Q, which
            # made the complex branch below unreachable and sent every dataset
            # through the interleaved-I/Q path.
            any_complex = any(np.iscomplexobj(a) for a in X_list)
            buf_dtype = np.complex64 if any_complex else np.float32
            X = np.zeros((len(X_list), target_len), dtype=buf_dtype)
            for i, a in enumerate(X_list):
                L = min(len(a), target_len)
                X[i, :L] = a[:L]

        y = np.asarray(y_list, dtype=np.int64)

        # Shape according to what the data actually is, not what the model is
        # named.  Complex baseband carries real quadrature and becomes two
        # channels; a real passband recording is one channel — its consecutive
        # samples are not I and Q, so splitting them would be meaningless.
        if is_multi_channel:
            pass  # already shaped as (batch, num_ch, target_len)
        elif np.iscomplexobj(X):
            X = self._prepare_iq_data(X)
            X = self._normalize_iq(X)
            input_channels = 2
            print("IQ mode: complex baseband -> 2 channels (I, Q)")
        else:
            X = self._prepare_1ch_data(X.astype(np.float32))
            input_channels = 1
            if self._is_iq_model:
                print(f"Note: {self.model_name} is an IQ architecture but the data "
                      "is real (passband); training it with 1 input channel. "
                      "Generate baseband (Complex IQ) datasets to use both.")

        # Shuffle
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Split train/val
        split = int(len(X) * (1.0 - self.val_split))
        if split < 1:
            split = max(1, len(X) - 1)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Convert to tensors (keep on CPU initially for pinned memory transfer)
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_val = torch.from_numpy(X_val)
        y_val = torch.from_numpy(y_val)

        # Create DataLoaders with pin_memory if using GPU
        use_pin = (self.device.type == 'cuda')
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  pin_memory=use_pin)

        val_ds = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                                pin_memory=use_pin)

        # Build model
        num_classes = len(self.labels)
        signal_len = X.shape[2]  # length after channel split
        model = get_model(self.model_name, num_classes=num_classes, input_size=signal_len,
                          in_channels=input_channels, plugin_path=self.model_plugin_path, **self.model_hparams)
        model.to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Scheduler & Scaler
        warmup_epochs = min(3, max(1, int(self.epochs * 0.15)))
        use_amp = (self.device.type == 'cuda')
        scaler = GradScaler('cuda', enabled=use_amp)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.epochs - warmup_epochs),
            eta_min=self.lr * 0.01
        )

        # Training loop.  epochs_done separates "finished or was stopped" from
        # "blew up", because only the first is worth saving.
        epochs_done = 0
        train_error = None
        try:
            for epoch in range(self.epochs):
                if self._stop:
                    print("Training stopped by user.")
                    break

                # Warmup logic
                if epoch < warmup_epochs:
                    curr_lr = self.lr * (epoch + 1) / warmup_epochs
                    for pg in optimizer.param_groups:
                        pg['lr'] = curr_lr
                else:
                    # Scheduler step is usually done at end of epoch, but notebook did it at start
                    # We'll follow notebook logic or standard PyTorch pattern (step at end usually)
                    # Notebook logic: else: scheduler.step() inside loop
                    scheduler.step()

                # Train
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for X_batch, y_batch in train_loader:
                    # Non-blocking transfer
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    
                    # Mixed Precision Forward
                    with autocast('cuda', enabled=use_amp):
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                    
                    # Mixed Precision Backward
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item() * X_batch.size(0)
                    _, pred = outputs.max(1)
                    train_correct += pred.eq(y_batch).sum().item()
                    train_total += y_batch.size(0)

                train_loss /= max(train_total, 1)
                train_acc = train_correct / max(train_total, 1)

                # Validate
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device, non_blocking=True)
                        y_batch = y_batch.to(self.device, non_blocking=True)
                        
                        with autocast('cuda', enabled=use_amp):
                            outputs = model(X_batch)
                            loss = criterion(outputs, y_batch)
                            
                        val_loss += loss.item() * X_batch.size(0)
                        _, pred = outputs.max(1)
                        val_correct += pred.eq(y_batch).sum().item()
                        val_total += y_batch.size(0)

                val_loss /= max(val_total, 1)
                val_acc = val_correct / max(val_total, 1)

                # Emit progress
                self.progress.emit(epoch + 1, self.epochs, train_loss, val_loss, train_acc, val_acc)
                epochs_done = epoch + 1

        except Exception as e:
            train_error = e
            print(f"Training failed: {e}")
            traceback.print_exc()

        # A run that raised has weights that mean nothing — often the random
        # initialisation, if it died in the first batch.  Saving them anyway
        # produced a .pth with a full metadata sidecar and a green
        # "Complete - saved", which the Evaluate and Inference tabs then loaded
        # as a trained model.  Report the failure instead; the training tab
        # already handles an empty path as "no model saved".
        if train_error is not None or epochs_done == 0:
            reason = (f"{type(train_error).__name__}: {train_error}"
                      if train_error is not None else
                      "no epoch completed")
            self.error = reason
            print(f"Not saving a model: {reason}")
            self.finished.emit("")
            return

        # Save model + metadata
        out_dir = self.save_dir if self.save_dir else os.path.join(os.getcwd(), 'models')
        os.makedirs(out_dir, exist_ok=True)
        timestamp = int(time.time())
        save_path = os.path.join(out_dir, f"{self.model_name}_{timestamp}.pth")
        meta_path = os.path.join(out_dir, f"{self.model_name}_{timestamp}.json")
        try:
            torch.save(model.state_dict(), save_path)
            if self.model_plugin:
                plugin_filename = f"{self.model_name}_{timestamp}_plugin.py"
                shutil.copy2(self.model_plugin["path"], os.path.join(out_dir, plugin_filename))
                saved_plugin = {"filename": plugin_filename, "name": self.model_plugin["name"], "sha256": self.model_plugin["sha256"]}
            else:
                saved_plugin = None
            # Save companion metadata
            # Record where the training data came from.  Without this a saved
            # model is indistinguishable from any other with the same shape, so
            # two runs that differ only in their input folder -- exactly what a
            # channel-condition comparison is -- produce sidecars that cannot be
            # told apart afterwards.
            source_dirs = sorted({os.path.dirname(p) for p, _ in self.file_label_pairs})
            common_root = os.path.commonpath(source_dirs) if source_dirs else ""

            metadata = {
                "model_name": self.model_name,
                "class_labels": self.labels,
                "num_classes": num_classes,
                "input_channels": input_channels,
                "signal_length": signal_len,
                # Without these the tabs rebuild the model with library
                # defaults, so anything trained at a non-default base_filters
                # could never be loaded again — it failed with dozens of size
                # mismatches inside load_state_dict.
                "model_hparams": dict(self.model_hparams),
                "model_plugin": saved_plugin,
                "timestamp": timestamp,
                "training_data_root": common_root,
                "training_class_dirs": source_dirs,
                "num_training_files": len(self.file_label_pairs),
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
                "val_split": self.val_split,
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Failed to save model: {e}")
            save_path = ""

        self.finished.emit(save_path)