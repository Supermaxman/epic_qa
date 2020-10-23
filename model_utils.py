
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
import pytorch_lightning as pl


class QuestionAnsweringBert(pl.LightningModule):
	def __init__(self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total, gamma, adv_temp):
		super().__init__()
		# self.bert = BertModel.from_pretrained(pre_model_name)
		self.bert = AutoModelForSequenceClassification.from_pretrained(pre_model_name)
		self.config = self.bert.config
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warmup = lr_warmup
		self.updates_total = updates_total
		self.adv_temp = adv_temp
		self.gamma = gamma
		self.save_hyperparameters()

	def forward(self, input_ids, attention_mask, token_type_ids):
		# [batch_size * sample_size, 2]
		logits = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)[0]
		# subtract positive class logit score, add negative class logit score
		# convert from softmax logits to sigmoid score
		# [batch_size * sample_size]
		energies = logits[:, 0] - logits[:, 1]
		return energies, None

	def _energy_loss(self, energies):
		# from Rotate paper
		# https://arxiv.org/abs/1902.10197
		# [batch_size]
		pos_energies = energies[:, 0]
		# [batch_size, neg_size]
		neg_energies = energies[:, 1:]
		with torch.no_grad():
			neg_probs = nn.Softmax(dim=1)(self.adv_temp * -neg_energies)
		pos_loss = -nn.LogSigmoid()(self.gamma - pos_energies)
		neg_loss = -neg_probs * nn.LogSigmoid()(neg_energies - self.gamma)
		neg_loss = neg_loss.sum(dim=1)
		loss = pos_loss + neg_loss
		return pos_energies, neg_energies, neg_probs, loss

	def training_step(self, batch, batch_nb):
		energies, _ = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
		)
		sample_size = batch['sample_size']
		energies = energies.view(-1, sample_size)
		batch_size = energies.shape[0]
		neg_size = sample_size - 1
		pos_energies, neg_energies, neg_probs, loss = self._energy_loss(
			energies
		)

		loss = loss.mean()
		# [bsize, neg_size]
		correct_count = (pos_energies.unsqueeze(1) < neg_energies).float()
		# sum number pos is less than and then subtract from neg count to get index, add 1 for rank
		pos_rank = neg_size - correct_count.sum(axis=1) + 1
		mrr = (1 / pos_rank).mean()

		exp_acc = (neg_probs * correct_count).sum(dim=1).sum(dim=0) / batch_size
		uniform_acc = correct_count.sum(dim=1).sum(dim=0) / (batch_size * neg_size)

		self.log('train_loss', loss)
		self.log('train_uniform_acc', uniform_acc)
		self.log('train_exp_acc', exp_acc)
		self.log(f'train_mrr@{neg_size + 1}', mrr)
		result = {
			'loss': loss
		}
		return result

	def validation_step(self, batch, batch_nb):
		energies, _ = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
		)
		sample_size = batch['sample_size']
		energies = energies.view(-1, sample_size)
		batch_size = energies.shape[0]
		neg_size = energies.shape[1] - 1
		pos_energies, neg_energies, neg_probs, loss = self._energy_loss(
			energies
		)
		correct_count = (pos_energies.unsqueeze(1) < neg_energies).float()
		# sum number pos is less than and then subtract from neg count to get index, add 1 for rank
		pos_rank = neg_size - correct_count.sum(axis=1) + 1
		mrr = (1 / pos_rank).mean()

		exp_acc = (neg_probs * correct_count).sum(dim=1).sum(dim=0) / batch_size
		uniform_acc = correct_count.sum(dim=1).sum(dim=0) / (batch_size * neg_size)

		result = {
			'val_loss': loss.mean(),
			'val_batch_loss': loss,
			'val_batch_uniform_acc': uniform_acc,
			'val_batch_exp_acc': exp_acc,
			f'val_batch_mrr@{neg_size+1}': mrr,
			f'neg_size': neg_size,
		}

		return result

	def validation_epoch_end(self, outputs):
		neg_size = outputs[0]['neg_size']
		loss = torch.cat([x['val_batch_loss'] for x in outputs], dim=0).mean()
		uniform_acc = torch.stack([x['val_batch_uniform_acc'] for x in outputs], dim=0).mean()
		exp_acc = torch.stack([x['val_batch_exp_acc'] for x in outputs], dim=0).mean()
		mrr = torch.stack([x[f'val_batch_mrr@{neg_size+1}'] for x in outputs], dim=0).mean()

		self.log('val_loss', loss)
		self.log('val_uniform_acc', uniform_acc)
		self.log('val_exp_acc', exp_acc)
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
