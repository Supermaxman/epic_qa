
import os
import pytorch_lightning as pl
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.distributed as dist
from transformers import T5Config, T5ForConditionalGeneration


class T5QueryGenerator(pl.LightningModule):
	def __init__(
			self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total,
			max_output_length, top_k, num_samples,
			torch_cache_dir, predict_mode=False, predict_path=None):
		super().__init__()
		self.pre_model_name = pre_model_name
		self.torch_cache_dir = torch_cache_dir
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warmup = lr_warmup
		self.updates_total = updates_total
		self.max_output_length = max_output_length
		self.top_k = top_k
		self.num_samples = num_samples
		self.predict_mode = predict_mode
		self.predict_path = predict_path
		self.config = T5Config.from_pretrained('t5-base')
		self.t5 = T5ForConditionalGeneration.from_pretrained(
			pre_model_name,
			from_tf=True,
			config=self.config
		)
		self.save_hyperparameters()

	def forward(self, input_ids, attention_mask):
		# [bsize, num_samples, max_output_length]
		outputs = self.t5.generate(
			input_ids=input_ids,
			attention_mask=attention_mask,
			min_length=self.max_output_length,
			max_length=self.max_output_length,
			do_sample=True,
			top_k=self.top_k,
			num_return_sequences=self.num_samples,
		)
		return outputs

	def training_step(self, batch, batch_nb):
		raise NotImplementedError()

	def test_step(self, batch, batch_nb):
		return self._eval_step(batch, batch_nb, 'test')

	def validation_step(self, batch, batch_nb):
		return self._eval_step(batch, batch_nb, 'val')

	def _forward_step(self, batch, batch_nb):
		outputs = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask']
		)
		if not self.predict_mode:
			raise NotImplementedError()
		else:
			return outputs

	def _eval_step(self, batch, batch_nb, name):
		if not self.predict_mode:
			raise NotImplementedError()
		else:
			outputs = self._forward_step(batch, batch_nb)
			print(f'{outputs.shape}')
			# [bsize, num_samples, max_output_length]
			outputs = outputs.detach()
			print(f'{outputs.shape}')
			outputs = outputs.cpu()
			print(f'{outputs.shape}')
			outputs = outputs.numpy()
			print(f'{outputs.shape}')
			print()
			# list of [num_samples, max_output_length]
			outputs = [x for x in outputs]
			device_id = get_device_id()


			self.write_prediction_dict(
				{
					'id': batch['id'],
					'samples': outputs,
				},
				filename=os.path.join(self.predict_path, f'predictions-{device_id}.pt')
			)
			result = {
				f'{name}_id': batch['id']
			}

			return result

	def _eval_epoch_end(self, outputs, name):
		if not self.predict_mode:
			raise NotImplementedError()
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


def get_device_id():
	try:
		device_id = dist.get_rank()
	except AssertionError:
		if 'XRT_SHARD_ORDINAL' in os.environ:
			device_id = int(os.environ['XRT_SHARD_ORDINAL'])
		else:
			device_id = 0
	return device_id
