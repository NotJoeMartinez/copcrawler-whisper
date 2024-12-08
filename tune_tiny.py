import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperTokenizer, 
    WhisperProcessor, 
    WhisperFeatureExtractor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
) 
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

def main():
    fine_tuner = FineTuner()
    fine_tuner.run()

class FineTuner:
    def __init__(self):

        self.model_id = 'openai/whisper-tiny'
        self.out_dir = 'whisper_tiny_atco2_v2'
        self.epochs = 10
        
        # Reduce batch sizes to fit into MPS memory more easily
        self.batch_size = 8

        print("Loading feature extractor and tokenizer")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.model_id, language='English', task='transcribe')
        self.processor = WhisperProcessor.from_pretrained(self.model_id, language='English', task='transcribe')

        # Evaluation metrics
        self.metric = evaluate.load('wer')

    def run(self):
        print("Loading dataset")
        atc_dataset_train = load_dataset('notjoemartinez/ATCO2-ASR', split='train')
        atc_dataset_valid = load_dataset('notjoemartinez/ATCO2-ASR', split='validation')
        
        print("Preparing data")
        # Reduce the number of workers and processes 
        # to avoid overhead on a laptop environment.
        atc_dataset_train = atc_dataset_train.cast_column('audio', Audio(sampling_rate=16000))
        atc_dataset_valid = atc_dataset_valid.cast_column('audio', Audio(sampling_rate=16000))

        atc_dataset_train = atc_dataset_train.map(
            self.prepare_dataset, 
            num_proc=1 # reduce from 4 to 1
        )
        atc_dataset_valid = atc_dataset_valid.map(
            self.prepare_dataset, 
            num_proc=1
        )

        print("Loading model")
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        model.generation_config.task = 'transcribe'
        model.generation_config.forced_decoder_ids = None

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )

        # Detect MPS availability
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        model.to(device)

        # Adjust training arguments for MPS usage
        # Switch bf16 to False and fp16 to True for MPS acceleration
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.out_dir, 
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=2,  # increase accumulation if needed
            learning_rate=1e-5,
            warmup_steps=1000,
            fp16=False,
            bf16=False,
            num_train_epochs=self.epochs,
            evaluation_strategy='epoch', 
            logging_strategy='epoch',
            save_strategy='epoch',
            predict_with_generate=True,
            generation_max_length=225,
            report_to=['tensorboard'],
            load_best_model_at_end=True,
            metric_for_best_model='wer',
            greater_is_better=False,
            # Reduce number of workers to avoid overhead
            dataloader_num_workers=2,
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

        print("Training model")
        trainer.train()

        print("Saving model")
        model.save_pretrained(f"{self.out_dir}/best_model")
        self.tokenizer.save_pretrained(f"{self.out_dir}/best_model")
        self.processor.save_pretrained(f"{self.out_dir}/best_model")


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
