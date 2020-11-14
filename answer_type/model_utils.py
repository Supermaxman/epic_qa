
from transformers import AutoModelForSequenceClassification, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
import pytorch_lightning as pl
from abc import ABC, abstractmethod


class ATPBert(pl.LightningModule, ABC):
	def __init__(self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total, category_map, types_map, torch_cache_dir):
		super().__init__()
		self.pre_model_name = pre_model_name
		self.category_map = category_map
		self.types_map = types_map
		self.torch_cache_dir = torch_cache_dir
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warmup = lr_warmup
		self.updates_total = updates_total
		self.score_func = nn.Softmax(dim=-1)
		self.category_criterion = nn.CrossEntropyLoss(reduction='none')
		self.types_criterion = nn.BCEWithLogitsLoss(reduction='none')
		self.save_hyperparameters()

	@abstractmethod
	def forward(self, input_ids, attention_mask, token_type_ids):
		pass

	def _loss(self, cat_logits, cat_labels, type_logits, type_labels):
		cat_loss = self.category_criterion(
			cat_logits,
			cat_labels
		)
		type_loss = self.types_criterion(
			type_logits,
			type_labels
		)
		type_loss = type_loss.mean(dim=-1)
		loss = cat_loss + type_loss
		return loss

	def training_step(self, batch, batch_nb):
		loss, logits, prediction, correct_count, total_count, accuracy = self._forward_step(batch, batch_nb)

		loss = loss.mean()
		self.log('train_loss', loss)
		self.log('train_accuracy', accuracy)
		result = {
			'loss': loss
		}
		return result

	def test_step(self, batch, batch_nb):
		return self._eval_step(batch, batch_nb, 'test')

	def validation_step(self, batch, batch_nb):
		return self._eval_step(batch, batch_nb, 'val')

	def _forward_step(self, batch, batch_nb):
		category_logits, types_logits = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
		)
		category_labels = batch['labels']
		types_labels = batch['types']
		loss = self._loss(
			category_logits,
			category_labels,
			types_logits,
			types_labels
		)
		batch_size = category_labels.shape[0]
		prediction = category_logits.max(dim=1)[1]
		correct_count = (prediction.eq(category_labels)).float().sum()
		total_count = float(batch_size)
		accuracy = correct_count / batch_size
		return loss, category_labels, prediction, correct_count, total_count, accuracy

	def _eval_step(self, batch, batch_nb, name):
		loss, logits, prediction, correct_count, total_count, accuracy = self._forward_step(batch, batch_nb)

		result = {
			f'{name}_loss': loss.mean(),
			f'{name}_batch_loss': loss,
			f'{name}_batch_accuracy': accuracy,
			f'{name}_correct_count': correct_count,
			f'{name}_total_count': total_count,
		}

		return result

	def _eval_epoch_end(self, outputs, name):
		loss = torch.cat([x[f'{name}_batch_loss'] for x in outputs], dim=0).mean()
		correct_count = torch.stack([x[f'{name}_correct_count'] for x in outputs], dim=0).sum()
		total_count = sum([x[f'{name}_total_count'] for x in outputs])
		accuracy = correct_count / total_count
		self.log(f'{name}_loss', loss)
		self.log(f'{name}_accuracy', accuracy)

	def validation_epoch_end(self, outputs):
		self._eval_epoch_end(outputs, 'val')

	def test_epoch_end(self, outputs):
		self._eval_epoch_end(outputs, 'test')

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


class ATPBertFromLanguageModel(ATPBert):
	def __init__(self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total, category_map, types_map, torch_cache_dir=None):
		super().__init__(pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total, category_map, types_map, torch_cache_dir)
		self.bert = BertModel.from_pretrained(
			pre_model_name,
			cache_dir=torch_cache_dir
		)
		self.category_classifier = nn.Linear(
			self.bert.config.hidden_size,
			len(category_map)
		)
		self.types_classifier = nn.Linear(
			self.bert.config.hidden_size,
			len(types_map)
		)
		self.config = self.bert.config

	def forward(self, input_ids, attention_mask, token_type_ids):
		cls_embeddings = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)[0][:, 0]
		# [batch_size, 2]
		category_logits = self.category_classifier(cls_embeddings)
		types_logits = self.types_classifier(cls_embeddings)
		return category_logits, types_logits


# class ATPBertFromSequenceClassification(ATPBert):
# 	def __init__(self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total, category_map, types_map, torch_cache_dir=None):
# 		super().__init__(pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total, category_map, types_map, torch_cache_dir)
# 		self.bert = AutoModelForSequenceClassification.from_pretrained(
# 			pre_model_name,
# 			cache_dir=torch_cache_dir
# 		)
# 		self.config = self.bert.config
#
# 	def forward(self, input_ids, attention_mask, token_type_ids):
# 		# [batch_size, 2]
# 		logits = self.bert(
# 			input_ids,
# 			attention_mask=attention_mask,
# 			token_type_ids=token_type_ids
# 		)[0]
# 		scores = self.score_func(logits)
# 		return logits, scores