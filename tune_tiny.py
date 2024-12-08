# import huggingface_hub

from datasets import load_dataset, DatasetDict
from transformers import (
    WhisperTokenizer, 
    WhisperProcessor, 
    WhisperFeatureExtractor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
) 
from datasets import Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import evaluate
# huggingface_hub.login()

def main():
    fine_tuner = FineTuner()
    fine_tuner.run()

class FineTuner:
    def __init__(self):

        self.model_id = 'openai/whisper-tiny'
        self.out_dir = 'whisper_tiny_atco2_v2'
        self.epochs = 10
        self.batch_size = 32

        print("Loading feature extractor and tokenizer")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.model_id, language='English', task='transcribe')
        self.processor = WhisperProcessor.from_pretrained(self.model_id, language='English', task='transcribe')

        # Evaluation metrics
        self.metric = evaluate.load('wer')

    def run(self):

        print("Loading dataset")
        atc_dataset_train = load_dataset('jlvdoorn/atco2-asr-atcosim', split='train')
        atc_dataset_valid = load_dataset('jlvdoorn/atco2-asr-atcosim', split='validation')
        
        print("Preparing data")
        atc_dataset_train = atc_dataset_train.cast_column('audio', Audio(sampling_rate=16000))
        atc_dataset_valid = atc_dataset_valid.cast_column('audio', Audio(sampling_rate=16000))

        atc_dataset_train = atc_dataset_train.map(
            self.prepare_dataset, 
            num_proc=4
        )
        atc_dataset_valid = atc_dataset_valid.map(
            self.prepare_dataset, 
            num_proc=4
        )


        # whispter model
        print("Loading model")
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        model.generation_config.task = 'transcribe'
        model.generation_config.forced_decoder_ids = None

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )


        # Define tranining configuration

        model_id = self.model_id
        out_dir = self.out_dir
        epochs = self.epochs
        batch_size = self.batch_size

        training_args = Seq2SeqTrainingArguments(
            output_dir=out_dir, 
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1, 
            learning_rate=0.00001,
            warmup_steps=1000,
            bf16=True,
            fp16=False,
            num_train_epochs=epochs,
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            save_strategy='epoch',
            predict_with_generate=True,
            generation_max_length=225,
            report_to=['tensorboard'],
            load_best_model_at_end=True,
            metric_for_best_model='wer',
            greater_is_better=False,
            dataloader_num_workers=8,
            save_total_limit=2,
            lr_scheduler_type='constant',
            seed=42,
            data_seed=42
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=atc_dataset_train,
            eval_dataset=atc_dataset_valid,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        # train model
        print("Training model")
        trainer.train()

        print("Saving model")
        model.save_pretrained(f"{out_dir}/best_model")
        print("tokenizer.save_pretrained")
        self.tokenizer.save_pretrained(f"{out_dir}/best_model")
        print("processor.save_pretrained")
        self.processor.save_pretrained(f"{out_dir}/best_model")

        # !zip -r whisper_tiny_atco2_v2 whisper_tiny_atco2_v2



    def prepare_dataset(self, batch):
        audio = batch['audio']
        batch['input_features'] = self.feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_features[0]
        batch['labels'] = self.tokenizer(batch['text']).input_ids
        return batch



    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return {'wer': wer}



# data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': feature['input_features']} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')

        label_features = [{'input_ids': feature['labels']} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')

        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch['labels'] = labels

        return batch


if __name__ == "__main__":
    main()

