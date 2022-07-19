import torch
from megatron import mpu
from megatron.model.language_model import parallel_lm_logits

loss_obj = torch.nn.CrossEntropyLoss(reduction='none')

def loss_fn(output, labels, fp16_lm_cross_entropy):
    """
    Compute the loss for the model.
    """

    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output
    )

    if fp16_lm_cross_entropy:
        return mpu.vocab_parallel_cross_entropy(output, labels).transpose(0,1).contiguous()
    else:
        return mpu.vocab_parallel_cross_entropy(output.float(), labels).transpose(0,1).contiguous()



