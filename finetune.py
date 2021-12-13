import os
import re
import logging
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from datasets import load_metric

from w2v2utils import Wav2Vec2ForSpeechClassification, DataCollatorWithPadding

from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Config
from transformers import AdamW, get_scheduler

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# CONFIG ------------------------------
# MODEL_NAME = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
MODEL_NAME = "facebook/wav2vec2-base-960h"
# MODEL_NAME = None
DATA_PATH = './'
NUM_EPOCHS = 100
BATCH_SIZE = 2
GPUS = 2
LEARNING_RATE = 1e-5
NUM_WARMUP_STEPS = 500
POOLING_MODE = "mean" # "mean", "sum", "max"
ACCUMULATE_GRAD_BATCHES = 1
OUTPUT_DIR = './results/'
# CHECKPOINT = 'last.ckpt'
CHECKPOINT = None
TEST = False
# --------------------------------------

class Custom_GTZAN(Dataset):
    def __init__(self, data_path, mode='train'):
        
        if mode == 'train':
            train_dataset = torchaudio.datasets.GTZAN(data_path, download=True, subset='training')
            self.original_dataset = train_dataset
        elif mode == 'validation':
            val_dataset = torchaudio.datasets.GTZAN(data_path, download=True, subset='validation')
            self.original_dataset = val_dataset
        else:
            test_dataset = torchaudio.datasets.GTZAN(data_path, download=True, subset='testing')
            self.original_dataset = test_dataset

        self.label2id = {'blues': 0,
                        'classical': 1,
                        'country': 2,
                        'disco': 3,
                        'hiphop': 4,
                        'jazz': 5,
                        'metal': 6,
                        'pop': 7,
                        'reggae': 8,
                        'rock': 9}
    
    def __getitem__(self, n):
        item = self.original_dataset[n]
        resampled, sample_rate, label = self.data_resample(item)
        batch = {'input_values': resampled, 'labels': label}
        return batch

    def data_resample(self, data, sample_rate=16000):
        resampler = torchaudio.transforms.Resample(data[1], sample_rate)
        resampled = resampler(data[0])
        return (resampled, sample_rate, self.label2id[data[2]])
        
    def __len__(self):
        return len(self.original_dataset)

class W2V2Finetune(LightningModule):
    def __init__(self, 
                model_name=MODEL_NAME,
                data_path=DATA_PATH, 
                learning_rate=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                ):
        super().__init__()

        self.save_hyperparameters()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name if MODEL_NAME else "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres",
            feature_size=1, 
            padding_value=0.0, 
            do_normalize=True, 
            # return_attention_mask=True
        )
        self.data_collator = DataCollatorWithPadding(
            feature_extractor=self.feature_extractor, 
            padding=True
        )

        self.acc_metric = load_metric("accuracy")

        if MODEL_NAME:
            self.model = Wav2Vec2ForSpeechClassification.from_pretrained(
                MODEL_NAME,
                gradient_checkpointing=True,
                problem_type="single_label_classification",
                id2label={
                    "0": "blues",
                    "1": "classical",
                    "2": "country",
                    "3": "disco",
                    "4": "hiphop",
                    "5": "jazz",
                    "6": "metal",
                    "7": "pop",
                    "8": "reggae",
                    "9": "rock"
                },
                label2id={
                    "blues": 0,
                    "classical": 1,
                    "country": 2,
                    "disco": 3,
                    "hiphop": 4,
                    "jazz": 5,
                    "metal": 6,
                    "pop": 7,
                    "reggae": 8,
                    "rock": 9
                },
            )
        else:
            self.config = Wav2Vec2Config.from_pretrained(
                gradient_checkpointing=True,
            )
            self.model = Wav2Vec2ForSpeechClassification(self.config)

        self.model.pooling_mode = POOLING_MODE

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        # loss = outputs.loss
        loss = outputs['loss']

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        # loss = outputs.loss
        loss = outputs['loss']
        
        # pred_ids = np.argmax(outputs.logits.detach().cpu().numpy(), axis=-1)
        pred_ids = np.argmax(outputs['logits'].detach().cpu().numpy(), axis=-1)
        label_ids = batch['labels'].detach().cpu().numpy()
        
        acc = self.acc_metric.compute(predictions=pred_ids, references=label_ids)['accuracy']

        self.log("v_loss", loss, on_epoch=True, prog_bar=True)
        self.log("acc", acc, on_epoch=True, prog_bar=True)
        
        return {"v_loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        # loss = outputs.loss
        loss = outputs['loss']
        
        # pred_ids = np.argmax(outputs.logits.detach().cpu().numpy(), axis=-1)
        pred_ids = np.argmax(outputs['logits'].detach().cpu().numpy(), axis=-1)
        label_ids = batch['labels'].detach().cpu().numpy()
        
        acc = self.acc_metric.compute(predictions=pred_ids, references=label_ids)['accuracy']

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("acc", acc, on_epoch=True, prog_bar=True)
        
        return {"test_loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.1, verbose=True)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "v_loss",
            },
        }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = Custom_GTZAN(data_path=DATA_PATH, mode='train')
            self.eval_data = Custom_GTZAN(data_path=DATA_PATH, mode='validation')
        
        if stage == "test" or stage is None:
            self.test_data = Custom_GTZAN(data_path=DATA_PATH, mode='test')

    def train_dataloader(self):
        self.train_data = Custom_GTZAN(data_path=DATA_PATH, mode='train')
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, collate_fn=self.data_collator, num_workers=GPUS*4)

    def val_dataloader(self):
        self.eval_data = Custom_GTZAN(data_path=DATA_PATH, mode='validation')
        return DataLoader(self.eval_data, batch_size=self.hparams.batch_size+1, collate_fn=self.data_collator, num_workers=GPUS*4)

    def test_dataloader(self):
        self.test_data = Custom_GTZAN(data_path=DATA_PATH, mode='test')
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size+1, collate_fn=self.data_collator, num_workers=GPUS*4)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict

if __name__ == '__main__':
    if CHECKPOINT:
        model = W2V2Finetune.load_from_checkpoint(OUTPUT_DIR + CHECKPOINT)
    else:
        model = W2V2Finetune()

    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename='checkpoint-{epoch:02d}',
        verbose=True,
        save_last=True,
        save_top_k=3,
        monitor="v_loss",
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='v_loss',
        patience=15,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        gpus=GPUS,
        strategy="ddp",
        amp_backend="apex",
        amp_level="O2",
        precision=16,
        max_epochs=NUM_EPOCHS,
        sync_batchnorm=True,
        callbacks=[early_stopping, checkpoint_callback],
    )

    if TEST:
        trainer.test(ckpt_path=OUTPUT_DIR + CHECKPOINT)
    else:
        trainer.fit(model)

    