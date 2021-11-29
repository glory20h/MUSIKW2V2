import os
import re
import logging
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from datasets import load_metric

from xlsrutils import *

from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForSpeechClassification
from transformers import AdamW, get_scheduler, Adafactor

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Root Data Path
data_path = '../Jeonbuk_Univ/Alzheimer_segmented_Recognition'

# training config-----------------------
NUM_EPOCHS = 5
BATCH_SIZE = 3
# BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_WARMUP_STEPS = 500
OUTPUT_DIR = './jb-xlsr-large-alz+sch/'
CHECKPOINT = None
# --------------------------------------

class JBDataset(Dataset):
    def __init__(self, data_path, split, processor):
        self.split = split
        self.processor = processor
        self.data_path = data_path
        
        trn_path = os.path.join(data_path, f'{split}.trn')
        alz_lines = open(trn_path, encoding='utf-8').readlines()
        # self.lines = lines[:90]
        self.alz_lines = alz_lines
        self.alx_len = len(self.alz_lines)

        # Add Schizophrenia Dataset for Training
        trn_path = os.path.join('../Jeonbuk_Univ/Schizophrenia_segmented_Recognition', f'{split}.trn')
        sch_lines = open(trn_path, encoding='utf-8').readlines()
        self.sch_lines = sch_lines
        
    def __getitem__(self, index):
        if index < self.alx_len:
            waveform_path, text = self.alz_lines[index].rstrip('\n').split(' :: ')
            waveform, sampling_rate = torchaudio.load(os.path.join(self.data_path, waveform_path))
        else:
            index -= self.alx_len
            waveform_path, text = self.sch_lines[index].rstrip('\n').split(' :: ')
            waveform, sampling_rate = torchaudio.load(os.path.join('../Jeonbuk_Univ/Schizophrenia_segmented_Recognition', waveform_path))

        # waveform_path, text = self.alz_lines[index].rstrip('\n').split(' :: ')
        # waveform, sampling_rate = torchaudio.load(os.path.join(self.data_path, waveform_path))

        input_values = self.processor(waveform, sampling_rate=sampling_rate).input_values[0][0].tolist()

        text = re.sub('[\,\?\.\!\-\;\:\"\“\%\‘\”\�]', '', text)
        with self.processor.as_target_processor():
            labels = self.processor(text).input_ids

        batch = {'input_values': input_values, 'labels': labels}
        return batch
        
    def __len__(self):
        # return len(self.alz_lines)
        return len(self.alz_lines) + len(self.sch_lines)

class W2V2Finetune(LightningModule):
    def __init__(self, 
                data_path=data_path, 
                checkpoint=CHECKPOINT, 
                learning_rate=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                ):
        super().__init__()

        self.save_hyperparameters()

        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

        self.wer_metric = load_metric("wer")
        self.cer_metric = load_metric("cer")

        # Load XLSR Model
        self.model = Wav2Vec2ForSpeechClassification.from_pretrained(
            "m3hrdadfi/wav2vec2-base-100k-voxpopuli-gtzan-music",
            gradient_checkpointing=True,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer)
        )

        self.steps = 0
        self.best_eval_loss = 100

        if checkpoint is not None:
            self.resume(checkpoint)

    def speech_file_to_array_fn(path, sampling_rate):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate)
        speech

    def resume(self, checkpoint):
        if os.path.isfile(checkpoint):
            print(f"=> loading checkpoint '{checkpoint}'")
            checkpoint = torch.load(checkpoint)
            self.steps = checkpoint['steps']
            self.best_eval_loss = checkpoint['best_eval_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{checkpoint}' at steps {self.steps})")
        else:
            print(f"=> no checkpoint found at '{checkpoint}'")

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        try:
            outputs = self(**batch)
            # loss = outputs.loss
            loss = outputs['loss']
            # self.log("t_loss", loss, on_epoch=True, prog_bar=True)

        except Exception as e:
            print(batch['input_values'].shape[-1])
            print(batch['labels'])
            print(e)
            return None

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        # loss = outputs.loss
        loss = outputs['loss']
        
        # pred_ids = np.argmax(outputs.logits.detach().cpu().numpy(), axis=-1)
        pred_ids = np.argmax(outputs['logits'].detach().cpu().numpy(), axis=-1)
        pred_str = self.processor.batch_decode(pred_ids)

        label_ids = batch['labels'].detach().cpu().numpy()
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(label_ids, group_tokens=False)
        
        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)

        self.log("v_loss", loss, on_epoch=True, prog_bar=True)
        self.log("wer", wer, on_epoch=True, prog_bar=True)
        self.log("cer", cer, on_epoch=True, prog_bar=True)
        
        return {"v_loss": loss, "wer": wer, "cer": cer}

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        # loss = outputs.loss
        loss = outputs['loss']
        
        # pred_ids = np.argmax(outputs.logits.detach().cpu().numpy(), axis=-1)
        pred_ids = np.argmax(outputs['logits'].detach().cpu().numpy(), axis=-1)
        pred_str = self.processor.batch_decode(pred_ids)

        label_ids = batch['labels'].detach().cpu().numpy()
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(label_ids, group_tokens=False)
        
        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("wer", wer, on_epoch=True, prog_bar=True)
        self.log("cer", cer, on_epoch=True, prog_bar=True)
        
        return {"test_loss": loss, "wer": wer, "cer": cer}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # optimizer = Adafactor(self.parameters(), lr=self.hparams.learning_rate)
        # lr_scheduler = get_scheduler(
        #     "linear",
        #     optimizer=optimizer,
        #     num_warmup_steps=num_warmup_steps,
        #     num_training_steps=num_training_steps
        # )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, verbose=True)
        # return optimizer
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "v_loss",
            },
        }

    # def configure_optimizers(self):
    #     return FusedAdam(self.parameters())

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = JBDataset(self.hparams.data_path, 'train', self.processor)
            self.eval_data = JBDataset(self.hparams.data_path, 'eval', self.processor)
        
        if stage == "test" or stage is None:
            self.test_data = JBDataset(self.hparams.data_path, 'test_ctrl', self.processor)

    def train_dataloader(self):
        self.train_data = JBDataset(self.hparams.data_path, 'train', self.processor)
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, collate_fn=self.data_collator, num_workers=16)

    def val_dataloader(self):
        self.eval_data = JBDataset(self.hparams.data_path, 'eval', self.processor)
        return DataLoader(self.eval_data, batch_size=self.hparams.batch_size+1, collate_fn=self.data_collator, num_workers=16)

    def test_dataloader(self):
        self.test_data = JBDataset(self.hparams.data_path, 'test_ctrl', self.processor)
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size+1, collate_fn=self.data_collator, num_workers=16)

    # def test_dataloader(self):
    #     eval_data = JBDataset(self.hparams.data_path, 'test_pati', self.processor)
    #     return DataLoader(eval_data, batch_size=3, collate_fn=data_collator)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict

if __name__ == '__main__':
    model = W2V2Finetune()

    if CHECKPOINT:
        model = XLSRJB().load_from_checkpoint(CHECKPOINT)

    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename='checkpoint-epoch{epoch:02d}',
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
        gpus=4,
        # gpus=[2],
        strategy="ddp",
        # accelerator="cpu",
        # strategy=DeepSpeedPlugin(stage=3, offload_optimizer=True, offload_parameters=True),
        # strategy="deepspeed_stage_2_offload",
        # plugins = [DDPPlugin(find_unused_parameters=False)],
        amp_backend="apex",
        amp_level="O2",
        precision=16,
        # accumulate_grad_batches=4,
        sync_batchnorm=True,
        callbacks=[early_stopping, checkpoint_callback],
        # auto_scale_batch_size = "binsearch",
        # auto_lr_find=True
    )

    # trainer = Trainer(
    #     gpus=1,
    #     # gpus=[2],
    #     strategy="ddp",
    #     # accelerator="cpu",
    #     # strategy=DeepSpeedPlugin(stage=3, offload_optimizer=True, offload_parameters=True),
    #     # strategy="deepspeed_stage_2",
    #     amp_backend="apex",
    #     amp_level="O2",
    #     precision=16,
    #     accumulate_grad_batches=4,
    #     sync_batchnorm=True,
    #     callbacks=[checkpoint_callback, early_stopping],
    #     # auto_scale_batch_size = "binsearch",
    #     # auto_lr_find=True
    # )

    # trainer.tune(model)

    trainer.fit(model)

    # trainer.test()

    # Retrieve checkpoint after training
    # checkpoint_callback = ModelCheckpoint(dirpath="my/path/")
    # trainer = Trainer(callbacks=[checkpoint_callback])
    # trainer.fit(model)
    # checkpoint_callback.best_model_path