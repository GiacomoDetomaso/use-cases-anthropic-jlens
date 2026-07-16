import torch
import torch.nn.functional as F

class SecurityBreachException(Exception):
    pass

class JSpaceControlPlane:
    def __init__(self, model, tokenizer, pre_fitted_j_lens, target_layer, threshold=0.75):
        self.model = model
        self.tokenizer = tokenizer
        self.J_l = pre_fitted_j_lens  # Pre-computed offline Jacobian matrix for layer l
        self.W_u = model.get_output_embeddings().weight  # Model's vocabulary unembedding
        self.target_layer = target_layer
        self.threshold = threshold

        # Define high-risk internal tracking concepts
        self.flagged_tokens = ["injection", "bypass", "override", "exploit", "fake"]
        self.flagged_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in self.flagged_tokens]

    def register_guardrail(self):
        # Dynamically tap into the targeted mid-layer of the transformer
        layer = self.model.model.layers[self.target_layer]
        layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # h_l is the hidden state vector passing through the residual stream
        h_l = output[0]

        # Apply the Anthropic J-Lens equation: softmax(W_u · norm(J_l · h_l))
        projected = torch.matmul(h_l, self.J_l.T)
        normalized = F.layer_norm(projected, projected.shape[-1:])
        j_space_logits = torch.matmul(normalized, self.W_u.T)
        j_space_probs = F.softmax(j_space_logits, dim=-1)

        # Check if the model is silently thinking about malicious concepts
        for token_id in self.flagged_ids:
            max_prob_in_sequence = j_space_probs[:, -1, token_id].max().item()
            if max_prob_in_sequence > self.threshold:
                # CRITICAL: Raise exception to abort generation BEFORE text hits the UI
                raise SecurityBreachException(
                    f"Malicious latent intent detected: {self.tokenizer.decode(token_id)}"
                )
