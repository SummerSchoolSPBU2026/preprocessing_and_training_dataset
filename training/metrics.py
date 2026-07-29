import numpy as np
import evaluate

class Metrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

    def compute(self, eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        prediction_text = self.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )

        labels = np.where(
            labels == -100,
            self.tokenizer.pad_token_id,
            labels,
        )

        reference_text = self.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        wer = self.wer_metric.compute(
            references=reference_text,
            predictions=prediction_text,
        )

        cer = self.cer_metric.compute(
            references=reference_text,
            predictions=prediction_text,
        )

        return {"wer": wer, "cer": cer}
# потом в trainer.py создаю
# metrics = Metrics(tokenizer)
# и передаю в
# trainer = Seq2SeqTrainer(
#     ...,
#     compute_metrics=metrics.compute,
# )