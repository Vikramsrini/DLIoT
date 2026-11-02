"""
losses.py
─────────
Focal loss + knowledge distillation for a single-logit binary classifier.

y_true = [label, teacher_logit], y_pred = student logit.

  L = α · Focal(y, z_s) + (1 − α) · T² · KL(σ(z_t / T) ‖ σ(z_s / T))

with the KL taken between the two Bernoulli distributions.
"""

import keras
from keras import ops


def binary_focal(y, z, gamma: float = 2.0, alpha_pos: float = 0.5):
    ce = ops.binary_crossentropy(y, z, from_logits=True)
    p = ops.sigmoid(z)
    p_t = y * p + (1.0 - y) * (1.0 - p)
    a_t = y * alpha_pos + (1.0 - y) * (1.0 - alpha_pos)
    return a_t * ops.power(1.0 - p_t, gamma) * ce


@keras.saving.register_keras_serializable(package="trustedge")
class DistillationLoss(keras.losses.Loss):
    def __init__(self, alpha=0.5, temperature=4.0, gamma=2.0, alpha_pos=0.5,
                 name="distillation_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.alpha_pos = float(alpha_pos)

    def call(self, y_true, y_pred):
        y, t, z = y_true[:, 0], y_true[:, 1], y_pred[:, 0]
        focal = binary_focal(y, z, self.gamma, self.alpha_pos)
        if self.alpha >= 1.0:
            return focal
        T = self.temperature
        q = ops.sigmoid(t / T)
        # KL(q ‖ p) = BCE(q, p) − H(q); subtracting the teacher entropy keeps the
        # loss at 0 when the student matches the teacher (same gradient).
        kl = (ops.binary_crossentropy(q, z / T, from_logits=True)
              - ops.binary_crossentropy(q, t / T, from_logits=True))
        return self.alpha * focal + (1.0 - self.alpha) * kl * T * T

    def get_config(self):
        return {**super().get_config(), "alpha": self.alpha, "temperature": self.temperature,
                "gamma": self.gamma, "alpha_pos": self.alpha_pos}


# Metrics read only the label column and turn logits into probabilities.
@keras.saving.register_keras_serializable(package="trustedge")
class LabelAUC(keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, :1], ops.sigmoid(y_pred), sample_weight)


@keras.saving.register_keras_serializable(package="trustedge")
class LabelAccuracy(keras.metrics.BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, :1], ops.sigmoid(y_pred), sample_weight)
