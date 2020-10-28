
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
import pytorch_lightning as pl


class QuestionAnsweringBert(pl.LightningModule):
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
		# [batch_size]
		pos_logits = logits[:, 0]
		pos_labels = labels[:, 0]
		# [batch_size, neg_size]
		neg_logits = logits[:, 1:]
		neg_labels = labels[:, 1:]

		pos_loss = self.criterion(
			pos_logits,
			pos_labels
		)
		neg_loss = self.criterion(
			neg_logits,
			neg_labels
		)
		neg_loss = neg_loss.mean(dim=1)
		loss = pos_loss + neg_loss
		return pos_logits, neg_logits, loss

	def training_step(self, batch, batch_nb):
		logits, _ = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
		)
		labels = batch['labels']
		sample_size = batch['sample_size']
		logits = logits.view(-1, sample_size, 2)
		batch_size = logits.shape[0]
		neg_size = sample_size - 1
		pos_logits, neg_logits, loss = self._loss(
			logits,
			labels
		)

		loss = loss.mean()
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
		sample_size = batch['sample_size']
		logits = logits.view(-1, sample_size, 2)
		batch_size = logits.shape[0]
		neg_size = logits.shape[1] - 1
		pos_logits, neg_logits, loss = self._energy_loss(
			logits
		)
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
