
from transformers import AutoModelForSequenceClassification, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
import pytorch_lightning as pl
from abc import ABC, abstractmethod


class RQEBert(pl.LightningModule, ABC):
	def __init__(self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total):
		super().__init__()
		self.pre_model_name = pre_model_name
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warmup = lr_warmup
		self.updates_total = updates_total
		self.score_func = nn.Softmax(dim=-1)
		self.criterion = nn.CrossEntropyLoss(reduction='none')
		self.save_hyperparameters()

	@abstractmethod
	def forward(self, input_ids, attention_mask, token_type_ids):
		pass

	def _loss(self, logits, labels):
		loss = self.criterion(
			logits,
			labels
		)

		return loss

	def training_step(self, batch, batch_nb):
		logits, _ = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
		)
		labels = batch['labels']
		loss = self._loss(
			logits,
			labels
		)
		loss = loss.mean()
		batch_size = logits.shape[0]
		prediction = logits.max(dim=1)[1]
		correct_count = (prediction.eq(labels)).float().sum()
		accuracy = correct_count.sum(dim=0) / batch_size

		self.log('train_loss', loss)
		self.log('train_accuracy', accuracy)
		result = {
			'loss': loss
		}
		return result

	def validation_step(self, batch, batch_nb):
		logits, _ = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
		)
		labels = batch['labels']
		loss = self._loss(
			logits,
			labels
		)
		batch_size = logits.shape[0]
		prediction = logits.max(dim=1)[1]
		correct_count = (prediction.eq(labels)).float().sum()
		accuracy = correct_count.sum(dim=0) / batch_size

		result = {
			'val_loss': loss.mean(),
			'val_batch_loss': loss,
			'val_batch_accuracy': accuracy
		}

		return result

	def validation_epoch_end(self, outputs):
		loss = torch.cat([x['val_batch_loss'] for x in outputs], dim=0).mean()
		accuracy = torch.stack([x['val_batch_accuracy'] for x in outputs], dim=0).mean()

		self.log('val_loss', loss)
		self.log('val_accuracy', accuracy)

	def configure_optimizers(self):
		params = self._get_optimizer_params(self.weight_decay)
		optimizer = AdamW(
			params,
			lr=self.learning_rate,
			weight_decay=self.weight_decay,
			correct_bias=False
		)
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=self.lr_warmup * self.updates_total,
			num_training_steps=self.updates_total
		)
		return [optimizer], [scheduler]

	def _get_optimizer_params(self, weight_decay):
		param_optimizer = list(self.named_parameters())
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_params = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
			 'weight_decay': weight_decay},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

		return optimizer_params


class RQEBertFromSequenceClassification(RQEBert):
	def __init__(self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total):
		super().__init__(pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total)
		self.bert = AutoModelForSequenceClassification.from_pretrained(
			pre_model_name,
			cache_dir='/users/max/data/models/torch_cache'
		)
		self.config = self.bert.config

	def forward(self, input_ids, attention_mask, token_type_ids):
		# [batch_size, 2]
		logits = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)[0]
		scores = self.score_func(logits)
		return logits, scores


class RQEBertFromLanguageModel(RQEBert):
	def __init__(self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total):
		super().__init__(pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total)
		self.bert = BertModel.from_pretrained(
			pre_model_name,
			cache_dir='/users/max/data/models/torch_cache'
		)
		self.classifier = nn.Linear(
			self.bert.config.hidden_size,
			2
		)
		self.config = self.bert.config

	def forward(self, input_ids, attention_mask, token_type_ids):
		cls_embeddings = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)[0][:, 0]
		# [batch_size, 2]
		logits = self.classifier(cls_embeddings)
		scores = self.score_func(logits)
		return logits, scores
