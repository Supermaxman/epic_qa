
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
import pytorch_lightning as pl


class QuestionAnsweringBert(pl.LightningModule):
	def __init__(self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total):
		super().__init__()
		# self.bert = BertModel.from_pretrained(pre_model_name)
		self.bert = AutoModelForSequenceClassification.from_pretrained(pre_model_name)
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warmup = lr_warmup
		self.updates_total = updates_total
		self.criterion = nn.CrossEntropyLoss(reduction='none')
		self.score_func = nn.Softmax(dim=-1)
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

		# [batch_size, sample_size]
		loss = self.criterion(logits, labels)
		# [batch_size]
		loss = loss.sum(dim=1)

		return loss

	def training_step(self, batch, batch_nb):
		logits, scores = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
		)
		logits = logits.view(-1, batch['sample_size'], 2)
		scores = scores.view(-1, batch['sample_size'], 2)
		labels = batch['labels'].view(-1, batch['sample_size'])
		loss = self._loss(
			logits,
			labels
		)

		loss = loss.mean()

		pos_scores = scores[:, 0, 1]
		neg_scores = scores[:, 1, 1]
		correct_count = (pos_scores > neg_scores).float()

		uniform_acc = correct_count.mean()

		result = {
			'loss': loss,
			'log': {
				'train_loss': loss,
				'train_uniform_acc': uniform_acc
			}
		}
		return result

	def validation_step(self, batch, batch_nb):
		logits, scores = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
		)
		logits = logits.view(-1, batch['sample_size'], 2)
		scores = scores.view(-1, batch['sample_size'], 2)
		labels = batch['labels'].view(-1, batch['sample_size'])
		loss = self._loss(
			logits,
			labels
		)

		pos_scores = scores[:, 0, 1]
		neg_scores = scores[:, 1, 1]
		correct_count = (pos_scores > neg_scores).float()

		uniform_acc = correct_count.mean()
		# TODO calculate proper metrics
		result = {
			'val_loss': loss.mean(),
			'val_batch_loss': loss,
			'val_batch_uniform_acc': uniform_acc
		}

		return result

	def validation_end(self, outputs):
		loss = torch.cat([x['val_batch_loss'] for x in outputs], dim=0).mean()
		uniform_acc = torch.stack([x['val_batch_uniform_acc'] for x in outputs], dim=0).mean()

		result = {
			'val_loss': loss,
			'val_uniform_acc': uniform_acc,
			'log': {
				'val_loss': loss,
				'val_uniform_acc': uniform_acc
			}
		}
		return result

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
