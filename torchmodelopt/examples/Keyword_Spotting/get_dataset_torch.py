# Handle Imports
import os
import shutil
import numpy as np
import functools
import torch
import torchaudio
import torchvision.transforms as transforms
import torchaudio.transforms as AT
import torch.nn.functional as F
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import kws_util
import torch_model as models

def get_audio_preprocessor(model_settings, is_training=False, background_data=[], device=torch.device("cpu")):
    def preprocess(audio_wave):
        desired_samples = model_settings['desired_samples']
        background_frequency = model_settings['background_frequency']
        background_volume_range_= model_settings['background_volume_range_']
        audio_wave = torch.squeeze(audio_wave)
        audio_wave = audio_wave.to(torch.float32)
        if model_settings['feature_type'] != "td_samples":
            audio_wave = audio_wave / torch.max(audio_wave).item()
        else:
            audio_wave = audio_wave / 2 ** 15
        #Previously, decode_wav was used with desired_samples as the length of array. The
        # default option of this function was to pad zeros if the desired samples are not found
        audio_wave = F.pad(audio_wave, (0, desired_samples - audio_wave.shape[-1]))
        
        # Allow the audio sample's volume to be adjusted.
        foreground_volume_ = 1.0
        
        scaled_foreground = audio_wave * foreground_volume_
        
        # Shift the sample's start position, and pad any gaps with zeros.
        time_shift_padding_ = (2, 2)
        time_shift_offset_ = 2
        padded_foreground = F.pad(scaled_foreground, time_shift_padding_, mode="constant", value=0)
        sliced_foreground = padded_foreground[time_shift_offset_ : time_shift_offset_ + desired_samples]

        # Add bg noise if applicable
        if is_training and background_data != []:
            background_volume_range = torch.tensor(background_volume_range_, dtype=torch.float32)
            background_index = np.random.randint(len(background_data))
            background_samples = background_data[background_index]
            background_offset = np.random.randint(0, len(background_samples) - desired_samples)
            background_clipped = background_samples[background_offset : (background_offset + desired_samples)]
            background_clipped = torch.squeeze(background_clipped)
            background_reshaped = F.pad(background_clipped, (0, desired_samples - audio_wave.shape[-1]))
            background_reshaped = background_reshaped.to(torch.float32)
            if np.random.uniform(0, 1) < background_frequency:
                background_volume = np.random.uniform(0, background_volume_range_)
            else:
                background_volume = 0
            background_mul = background_reshaped * background_volume
            background_add = background_mul + sliced_foreground
            sliced_foreground = torch.clamp(background_add, -1.0, 1.0)

        # Calculate MFCC if applicable (Use TF for consistency)
        if model_settings['feature_type'] == 'mfcc':
            if model_settings["framework"] == "torch":
                mfcc_ = AT.MFCC(
                    sample_rate=model_settings["sample_rate"],
                    n_mfcc=model_settings["dct_coefficient_count"],
                    melkwargs={
                        "n_fft": int(16000 * 40 / 1000),
                        "hop_length": int(16000 * 20/ 1000),
                        "n_mels": 128,
                        "mel_scale": "htk",
                        "center": False 
                    }
                ).to(device)
                mfccs = mfcc_(sliced_foreground).T.unsqueeze(dim=0)
                assert mfccs.shape == torch.Size([1, 49, 10]), "SHAPE ERROR"
                return mfccs
            else:
                # Convert to TF Tensor
                sliced_foreground_tf = tf.convert_to_tensor(sliced_foreground.numpy(), dtype=tf.float32)
                stfts = tf.signal.stft(sliced_foreground_tf, frame_length=model_settings['window_size_samples'], 
                            frame_step=model_settings['window_stride_samples'], fft_length=None,
                            window_fn=tf.signal.hann_window
                            )
                spectrograms = tf.abs(stfts)
                num_spectrogram_bins = stfts.shape[-1]
                # default values used by contrib_audio.mfcc as shown here
                # https://kite.com/python/docs/tensorflow.contrib.slim.rev_block_lib.contrib_framework_ops.audio_ops.mfcc
                lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40 
                linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix( num_mel_bins, num_spectrogram_bins,
                                                                                    model_settings['sample_rate'],
                                                                                    lower_edge_hertz, upper_edge_hertz)
                mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
                mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
                # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
                log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
                # Compute MFCCs from log_mel_spectrograms and take the first 13.
                mfccs_tf = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :model_settings['dct_coefficient_count']]
                mfccs_tf = tf.reshape(mfccs_tf, [1, model_settings['spectrogram_length'], model_settings['dct_coefficient_count']])
                mfccs = torch.tensor(mfccs_tf.numpy(), dtype=torch.float32)
                audio_wave = mfccs
        # Calculate LFBE if applicable
        elif model_settings['feature_type'] == "lfbe":
            # Convert to TF
            sliced_foreground_tf = tf.convert_to_tensor(sliced_foreground.numpy(), dtype=tf.float32)

            # apply preemphasis
            preemphasis_coef = 1 - 2 ** -5
            power_offset = 52
            num_mel_bins = model_settings['dct_coefficient_count']
            paddings = tf.constant([[0, 0], [1, 0]])

            # for some reason, tf.pad only works with the extra batch dimension, but then we remove it after pad
            sliced_foreground_tf = tf.expand_dims(sliced_foreground_tf, 0)
            sliced_foreground_tf = tf.pad(tensor=sliced_foreground_tf, paddings=paddings, mode='CONSTANT')
            sliced_foreground_tf = sliced_foreground_tf[:, 1:] - preemphasis_coef * sliced_foreground_tf[:, :-1]
            sliced_foreground_tf = tf.squeeze(sliced_foreground_tf) 

            # compute fft
            stfts = tf.signal.stft(sliced_foreground_tf,  frame_length=model_settings['window_size_samples'], 
                                    frame_step=model_settings['window_stride_samples'], fft_length=None,
                                    window_fn=functools.partial(
                                    tf.signal.hamming_window, periodic=False),
                                    pad_end=False,
                                    name='STFT')

            # compute magnitude spectrum [batch_size, num_frames, NFFT]
            magspec = tf.abs(stfts)
            num_spectrogram_bins = magspec.shape[-1]

            # compute power spectrum [num_frames, NFFT]
            powspec = (1 / model_settings['window_size_samples']) * tf.square(magspec)
            powspec_max = tf.reduce_max(input_tensor=powspec)
            powspec = tf.clip_by_value(powspec, 1e-30, powspec_max) # prevent -infinity on log

            def log10(x):
                # Compute log base 10 on the tensorflow graph.
                # x is a tensor.  returns log10(x) as a tensor
                numerator = tf.math.log(x)
                denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
                return numerator / denominator

            # Warp the linear-scale, magnitude spectrograms into the mel-scale.
            lower_edge_hertz, upper_edge_hertz = 0.0, model_settings['sample_rate'] / 2.0
            linear_to_mel_weight_matrix = (
                tf.signal.linear_to_mel_weight_matrix(
                    num_mel_bins=num_mel_bins,
                    num_spectrogram_bins=num_spectrogram_bins,
                    sample_rate=model_settings['sample_rate'],
                    lower_edge_hertz=lower_edge_hertz,
                    upper_edge_hertz=upper_edge_hertz))

            mel_spectrograms = tf.tensordot(powspec, linear_to_mel_weight_matrix,1)
            mel_spectrograms.set_shape(magspec.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]))

            log_mel_spec_tf = 10 * log10(mel_spectrograms)
            log_mel_spec_tf = tf.expand_dims(log_mel_spec_tf, 0, name="mel_spec")

            log_mel_spec_tf = (log_mel_spec_tf + power_offset - 32 + 32.0) / 64.0
            log_mel_spec_tf = tf.clip_by_value(log_mel_spec_tf, 0, 1)

            # Convert back to torch
            log_mel_spec = torch.tensor(log_mel_spec_tf.numpy(), dtype=torch.float32)
            audio_wave = log_mel_spec

        # Else Return Time_Domain Features
        elif model_settings['feature_type'] == 'td_samples':
            ## sliced_foreground should have the right data.  Make sure it's the right format (int16)
            # and just return it.
            paddings = (0, 16000 - sliced_foreground.shape[0])
            wav_padded = F.pad(sliced_foreground, paddings)
            wav_padded = wav_padded.unsqueeze(-1)
            wav_padded = wav_padded.unsqueeze(0)
            audio_wave = wav_padded

        return audio_wave

    return preprocess

def prepare_background_data(bg_path,BACKGROUND_NOISE_DIR_NAME, device):
  """Searches a folder for background noise audio, and loads it into memory.
  It's expected that the background audio samples will be in a subdirectory
  named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
  the sample rate of the training data, but can be much longer in duration.
  If the '_background_noise_' folder doesn't exist at all, this isn't an
  error, it's just taken to mean that no background noise augmentation should
  be used. If the folder does exist, but it's empty, that's treated as an
  error.
  Returns:
    List of raw PCM-encoded audio samples of background noise.
  Raises:
    Exception: If files aren't found in the folder.
  """
  background_data = []
  background_dir = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME)
  if not os.path.exists(background_dir):
    return background_data
  for wav in os.listdir(background_dir):
    wav_path = os.path.join(background_dir, wav)
    if not wav_path.endswith((".wav", ".WAV")):
        continue
    audio, _ = torchaudio.load(wav_path)
    background_data.append(audio.squeeze().to(device))
  if not background_data:
    raise Exception('No background wav files were found in ' + background_dir)
  return background_data

class CustomAudioDataset(Dataset):
    def __init__(self, audio_wave, label, preprocessor = None, cast_and_pad = False):
        self.audio_wave = audio_wave
        self.label = label
        if preprocessor is not None:
           self.audio_wave = preprocessor(self.audio_wave)
        if cast_and_pad:
            paddings = (0, 16000 - audio_wave.shape[0])
            self.audio_wave = F.pad(audio_wave, paddings)
            self.audio_wave = self.audio_wave.to(torch.int16)
    def __len__(self):
        return len(self.audio_wave)
    
    def __getitem__(self, idx):
        return {"audio" : self.audio_wave[idx], "label" : self.label[idx]}

class SavedTensorDataset(Dataset):
    def __init__(self, dataset_dir):
        self.encode_labels = {
            "down": 0,
            "go": 1,
            "left": 2,
            "no": 3,
            "off": 4,
            "on": 5,
            "right": 6,
            "stop": 7,
            "up": 8,
            "yes": 9,
            "_silence_": 10,
            "_unknown_": 11,
        }
        self.max = 85511 if "train" in dataset_dir else 4890 if "test" in dataset_dir else 10102
        self.tensors = []
        self.labels = []
        self.filenames = []
        self.class_count=torch.zeros(12)
        with tqdm(total=self.max, desc="loading Tensor...", unit='tensor') as pbar:
            for label in os.listdir(dataset_dir):
                if label not in self.encode_labels.keys():
                    continue
                label_dir = os.path.join(dataset_dir, label)
                if os.path.isdir(label_dir):
                    for tensor_file in os.listdir(label_dir):
                        audio_tensor = torch.load(os.path.join(label_dir, tensor_file), weights_only=True)
                        self.tensors.append(audio_tensor)
                        self.labels.append(label)
                        self.class_count[self.encode_labels[label]] += 1
                        pbar.update(1)
                        self.filenames.append(os.path.join(label_dir, tensor_file))  # Store full path to file
        self.labels = [self.encode_labels[label] for label in self.labels]
        
        self.class_weights = self.class_count.sum().item() / (12 * self.class_count)

    def __len__(self):
        return len(self.labels)
    
    def _get_imbalance(self):
        return self.class_weights
    
    def __getitem__(self, index):
        audio_tensor = self.tensors[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)  # Convert label to tensor
        filename = self.filenames[index]  # Include filename
        return {"audio" : audio_tensor, "label" : label}


                    

class GoogleSpeechDataset(Dataset):
    def __init__(self, dataset_dir, model_settings, is_training=False, background_data=[], need_raw=False, cast_and_pad=False, save_tensor=False, save_dir=None):
        super().__init__()
        self.num_classes = 12
        self.device = torch.device("cpu")
        self.dataset_dir = dataset_dir
        self.model_settings = model_settings
        self.is_training = is_training
        self.need_raw = need_raw
        self.cast_and_pad = cast_and_pad
        self.bg_data = background_data
        self.max = 85511 if "train" in dataset_dir else 4890 if "test" in dataset_dir else 10102
        self.preprocess = get_audio_preprocessor(model_settings, is_training, background_data, self.device)
        self.audio = []
        self.labels = []
        self.encode_labels = {
            "down": 0,
            "go": 1,
            "left": 2,
            "no": 3,
            "off": 4,
            "on": 5,
            "right": 6,
            "stop": 7,
            "up": 8,
            "yes": 9,
            "_silence_": 10,
            "_unknown_": 11,
        }
        self.count = 0
        self.class_count = torch.zeros(self.num_classes)
        with tqdm(total=self.max, unit='file') as pbar:
            for label in os.listdir(dataset_dir):
                if label not in self.encode_labels.keys():
                    continue
                label_dir = os.path.join(dataset_dir, label)
                if os.path.isdir(label_dir):
                    for audio_file in os.listdir(label_dir):
                        if audio_file.endswith(('.wav', '.WAV')):
                            # if self.count == 100:
                            #     break
                            raw_audio_wav, _ = torchaudio.load(os.path.join(label_dir, audio_file))
                            raw_audio_wav = raw_audio_wav.to(self.device)
                            if self.cast_and_pad:
                                raw_audio_wav = F.pad(raw_audio_wav, (0, 16000 - raw_audio_wav.shape[-1]))
                                raw_audio_wav = raw_audio_wav.to(dtype=torch.int16)
                            audio_wav = raw_audio_wav if self.need_raw else self.preprocess(raw_audio_wav)
                            self.audio.append(audio_wav)
                            self.labels.append(label)
                            self.class_count[self.encode_labels[label]] += 1
                            if save_tensor:
                                save_label_dir = os.path.join(save_dir, label)
                                if not os.path.isdir(save_label_dir):
                                    os.mkdir(save_label_dir)
                                torch.save(audio_wav, os.path.join(save_label_dir, f"{audio_file}_tensor.pt"))
                            # self.count += 1
                            pbar.update(1)
        self.class_weights = self.class_count.sum().item() / (12 * self.class_count)
        self.labels = [self.encode_labels[label] for label in self.labels]
        

    def __len__(self):
        return len(self.audio)
    
    def _get_imbalance(self):
        return self.class_weights

    def __getitem__(self, index):
        audio_wav = self.audio[index]
        label = self.labels[index]
        return {"audio" : audio_wav, "label" : label}

def get_training_data(Flags, get_waves=False, val_cal_subset=False, device=torch.device("cpu")):

    label_count=12
    background_frequency = Flags.background_frequency
    background_volume_range_= Flags.background_volume
    model_settings = models.prepare_model_settings(label_count, Flags)

    bg_path=Flags.bg_path
    BACKGROUND_NOISE_DIR_NAME='_background_noise_' 
    background_data = prepare_background_data(bg_path, BACKGROUND_NOISE_DIR_NAME, device)
    background_data = []
    data_dir =  Flags.data_dir if Flags.save_tensor or not Flags.load_tensor else Flags.save_dir
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    val_dir = os.path.join(data_dir, "val")

    if Flags.save_tensor:
        if os.path.isdir(Flags.save_dir):
            shutil.rmtree(Flags.save_dir)
        os.mkdir(Flags.save_dir)
        os.mkdir(os.path.join(Flags.save_dir, "train"))
        os.mkdir(os.path.join(Flags.save_dir, "test"))
        os.mkdir(os.path.join(Flags.save_dir, "val"))


    ds_val = None
    if val_cal_subset:  # only return the subset of val set used for quantization calibration
        ds_val = SavedTensorDataset(dataset_dir=val_dir)
        print(os.path.join(data_dir, "val"))
        with open("quant_cal_idxs.txt") as fpi:
            cal_indices = [int(line) for line in fpi]
            print(cal_indices)
        cal_indices.sort()
        # cal_indices are the positions of specific inputs that are selected to calibrate the quantization
        count = 0  # count will be the index into the validation set.
        val_sub_audio = []
        val_sub_labels = []
        for d in ds_val:
            if count in cal_indices:          # this is one of the calibration inpus
                new_audio = d['audio']  
                if len(new_audio) < 16000:      # from_tensor_slices doesn't work for ragged tensors, so pad to 16k
                    new_audio = F.pad(new_audio, (0, 16000 - new_audio.shape[-1]), 'constant')
                val_sub_audio.append(new_audio)
                val_sub_labels.append(d['label'].numpy())
                count += 1
  #      if get_waves:
        ds_val = CustomAudioDataset(torch.tensor(val_sub_audio), torch.tensor(val_sub_labels), preprocessor=None, cast_and_pad=False)
   #     else:
    #        ds_val = CustomAudioDataset(torch.tensor(val_sub_audio), torch.tensor(val_sub_labels), preprocessor=get_audio_preprocessor(model_settings,
     #                                                                                                                               False, background_data))

    if get_waves :
        ds_train = GoogleSpeechDataset(dataset_dir=train_dir, model_settings=model_settings, 
                                    is_training=True, background_data=background_data, need_raw=True, cast_and_pad=True)
        ds_test = GoogleSpeechDataset(dataset_dir=test_dir, model_settings=model_settings, 
                                    is_training=False, background_data=background_data, need_raw=True, cast_and_pad=True)
        if not val_cal_subset:
            ds_val = GoogleSpeechDataset(
                dataset_dir=val_dir,
                model_settings=model_settings,
                is_training=False,
                background_data=background_data,
                need_raw=True, cast_and_pad=True
            )
    else:
        if Flags.save_tensor or not Flags.load_tensor:
            ds_train = GoogleSpeechDataset(
                dataset_dir=train_dir,
                model_settings=model_settings,
                is_training=True,
                background_data=background_data,
                save_tensor=Flags.save_tensor,
                save_dir=os.path.join(Flags.save_dir, "train")
            )
            ds_test = GoogleSpeechDataset(
                dataset_dir=test_dir,
                model_settings=model_settings,
                is_training=False,
                background_data=background_data,
                save_tensor=Flags.save_tensor,
                save_dir=os.path.join(Flags.save_dir, "test")
            )
            if not val_cal_subset:
                ds_val = GoogleSpeechDataset(
                    dataset_dir=val_dir,
                    model_settings=model_settings,
                    is_training=False,
                    background_data=background_data,
                    save_tensor=Flags.save_tensor,
                    save_dir=os.path.join(Flags.save_dir, "val")
                )
        else:
            ds_train = SavedTensorDataset(dataset_dir=train_dir)
            ds_test = SavedTensorDataset(dataset_dir=test_dir)
            if not val_cal_subset:
                ds_val = SavedTensorDataset(dataset_dir=val_dir)



    if Flags.num_train_samples != -1:
        ds_train = Subset(ds_train, torch.randperm(len(ds_train))[:Flags.num_train_samples])
    if Flags.num_val_samples != -1:
        ds_val = Subset(ds_val, torch.randperm(len(ds_val))[:Flags.num_val_samples])
    if Flags.num_test_samples != -1:
        ds_test = Subset(ds_test, torch.randperm(len(ds_test))[: Flags.num_test_samples])

    # Now that we've acquired the preprocessed data, either by processing or loading,
    ds_train_loader = DataLoader(dataset=ds_train, batch_size=Flags.batch_size, shuffle=True)
    ds_test_loader = DataLoader(dataset=ds_test, batch_size=Flags.batch_size, shuffle=True)
    ds_val_loader = DataLoader(dataset=ds_val, batch_size=Flags.batch_size, shuffle=True)
    return ds_train_loader, ds_test_loader, ds_val_loader, ds_train._get_imbalance()
