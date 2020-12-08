
from transformers import AutoModelForSequenceClassification
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
import pytorch_lightning as pl
from abc import ABC, abstractmethod
import torch.distributed as dist
import os


class QuestionAnsweringSampledBert(pl.LightningModule):
	def __init__(self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total, adv_temp):
		super().__init__()
		# self.bert = BertModel.from_pretrained(pre_model_name)
		self.bert = AutoModelForSequenceClassification.from_pretrained(pre_model_name)
		self.config = self.bert.config
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warmup = lr_warmup
		self.updates_total = updates_total
		self.adv_temp = adv_temp
		self.score_func = nn.Softmax(dim=-1)
		self.criterion = nn.CrossEntropyLoss(reduction='none')
		self.save_hyperparameters()

	def forward(self, input_ids, attention_mask, token_type_ids):
		# [batch_size * sample_size, 2]
		logits = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)[0]
		scores = self.score_func(logits)
		return logits, scores

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
		loss = self._loss(
			logits,
			batch['labels']
		)
		loss = loss.mean()
		sample_size = batch['sample_size']
		logits = logits.view(-1, sample_size, 2)
		pos_logits = logits[:, 0, 1]
		neg_logits = logits[:, 1:, 1]
		batch_size = logits.shape[0]
		neg_size = sample_size - 1

		# [bsize, neg_size]
		correct_count = (pos_logits.unsqueeze(1) > neg_logits).float()
		# sum number pos is less than and then subtract from neg count to get index, add 1 for rank
		pos_rank = neg_size - correct_count.sum(axis=1) + 1
		mrr = (1 / pos_rank).mean()

		# exp_acc = (neg_probs * correct_count).sum(dim=1).sum(dim=0) / batch_size
		uniform_acc = correct_count.sum(dim=1).sum(dim=0) / (batch_size * neg_size)

		self.log('train_loss', loss)
		self.log('train_uniform_acc', uniform_acc)
		# self.log('train_exp_acc', exp_acc)
		self.log(f'train_mrr@{neg_size + 1}', mrr)
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

		loss = self._loss(
			logits,
			batch['labels']
		)
		sample_size = batch['sample_size']
		logits = logits.view(-1, sample_size, 2)
		batch_size = logits.shape[0]
		neg_size = logits.shape[1] - 1
		pos_logits = logits[:, 0, 1]
		neg_logits = logits[:, 1:, 1]
		correct_count = (pos_logits.unsqueeze(1) > neg_logits).float()
		# sum number pos is less than and then subtract from neg count to get index, add 1 for rank
		pos_rank = neg_size - correct_count.sum(axis=1) + 1
		mrr = (1 / pos_rank).mean()

		# exp_acc = (neg_probs * correct_count).sum(dim=1).sum(dim=0) / batch_size
		uniform_acc = correct_count.sum(dim=1).sum(dim=0) / (batch_size * neg_size)

		result = {
			'val_loss': loss.mean(),
			'val_batch_loss': loss,
			'val_batch_uniform_acc': uniform_acc,
			# 'val_batch_exp_acc': exp_acc,
			f'val_batch_mrr@{neg_size+1}': mrr,
			f'neg_size': neg_size,
		}

		return result

	def validation_epoch_end(self, outputs):
		neg_size = outputs[0]['neg_size']
		loss = torch.cat([x['val_batch_loss'] for x in outputs], dim=0).mean()
		uniform_acc = torch.stack([x['val_batch_uniform_acc'] for x in outputs], dim=0).mean()
		# exp_acc = torch.stack([x['val_batch_exp_acc'] for x in outputs], dim=0).mean()
		mrr = torch.stack([x[f'val_batch_mrr@{neg_size+1}'] for x in outputs], dim=0).mean()

		self.log('val_loss', loss)
		self.log('val_uniform_acc', uniform_acc)
		# self.log('val_exp_acc', exp_acc)
		self.log(f'val_mrr@{neg_size+1}', mrr)

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


class RerankBert(pl.LightningModule):
	def __init__(
			self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total,
			torch_cache_dir, predict_mode=False, predict_path=None, weighted_loss=False):
		super().__init__()
		self.pre_model_name = pre_model_name
		self.torch_cache_dir = torch_cache_dir
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warmup = lr_warmup
		self.updates_total = updates_total
		self.predict_mode = predict_mode
		self.predict_path = predict_path
		self.weighted_loss = weighted_loss
		self.bert = AutoModelForSequenceClassification.from_pretrained(
			pre_model_name,
			cache_dir=torch_cache_dir
		)
		self.config = self.bert.config
		self.criterion = nn.CrossEntropyLoss(reduction='none')
		self.save_hyperparameters()

	def forward(self, input_ids, attention_mask, token_type_ids):
		# [batch_size, 2]
		logits = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)[0]
		return logits

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
		logits = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
		)
		if not self.predict_mode:
			labels = batch['labels']
			if self.weighted_loss:
				weights = batch['weights']
				loss = self._loss(
					logits,
					labels,
					weights
				)
			else:
				loss = self._loss(
					logits,
					labels,
				)
			# TODO add weights batch['weights']
			prediction = logits.max(dim=1)[1]
			correct_count = ((labels.eq(1)).float() * (prediction.eq(labels)).float()).sum()
			total_count = (labels.eq(1)).float().sum()
			accuracy = correct_count / total_count
			if accuracy.isnan().item():
				accuracy = torch.zeros(1, dtype=torch.float)

			return loss, logits, prediction, correct_count, total_count, accuracy
		else:
			return logits

	def _eval_step(self, batch, batch_nb, name):
		if not self.predict_mode:
			loss, logits, prediction, correct_count, total_count, accuracy = self._forward_step(batch, batch_nb)

			result = {
				f'{name}_loss': loss.mean(),
				f'{name}_batch_loss': loss,
				f'{name}_batch_accuracy': accuracy,
				f'{name}_correct_count': correct_count,
				f'{name}_total_count': total_count,
				f'{name}_batch_logits': logits,
				f'{name}_batch_labels': batch['labels'],
			}

			return result
		else:
			logits = self._forward_step(batch, batch_nb)
			logits = logits.detach()
			device_id = get_device_id()
			self.write_prediction_dict(
				{
					'id': batch['id'],
					'question_id': batch['question_id'],
					'pos_score': logits[:, 1].tolist(),
					'neg_score': logits[:, 0].tolist(),
				},
				filename=os.path.join(self.predict_path, f'predictions-{device_id}.pt')
			)
			result = {
				f'{name}_id': batch['id'],
				f'{name}_question_id': batch['question_id'],
				f'{name}_logits': logits,
			}

			return result

	def _eval_epoch_end(self, outputs, name):
		if not self.predict_mode:
			loss = torch.cat([x[f'{name}_batch_loss'] for x in outputs], dim=0).mean()
			logits = torch.cat([x[f'{name}_batch_logits'] for x in outputs], dim=0)
			labels = torch.cat([x[f'{name}_batch_labels'] for x in outputs], dim=0)
			score = logits[:, 1] - logits[:, 0]
			pos_mask = labels.eq(1).float()
			neg_mask = labels.eq(0).float()
			pos_avg_score = (pos_mask * score).sum() / pos_mask.sum()
			neg_avg_score = (neg_mask * score).sum() / neg_mask.sum()
			margin = pos_avg_score - neg_avg_score

			correct_count = torch.stack([x[f'{name}_correct_count'] for x in outputs], dim=0).sum()
			total_count = sum([x[f'{name}_total_count'] for x in outputs])
			accuracy = correct_count / total_count
			self.log(f'{name}_loss', loss)
			self.log(f'{name}_accuracy', accuracy)
			self.log(f'{name}_margin', margin)
		else:
			pass

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

	def _loss(self, logits, labels, weights=None):
		loss = self.criterion(
			logits,
			labels
		)
		if weights is not None:
			loss = loss * weights

		return loss


def get_device_id():
	try:
		device_id = dist.get_rank()
	except AssertionError:
		if 'XRT_SHARD_ORDINAL' in os.environ:
			device_id = int(os.environ['XRT_SHARD_ORDINAL'])
		else:
			device_id = 0
	return device_id
