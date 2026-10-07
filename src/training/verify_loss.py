"""Compare original and selective losses/LoRA gradients on a complete 1024-token sample."""

import json
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForMultimodalLM

from src.training.loss import assistant_loss
from src.training.m0_lora_check import prepare_samples


def main() -> None:
    config=json.loads(Path('configs/m0_training.json').read_text())
    model_config=yaml.safe_load(Path(config['model_config']).read_text())['model']
    torch.manual_seed(0)
    tokenizer=AutoTokenizer.from_pretrained(model_config['path'],local_files_only=True)
    samples,_=prepare_samples(config,tokenizer)
    sample=samples[0]
    base=AutoModelForMultimodalLM.from_pretrained(model_config['path'],
        dtype=torch.bfloat16,device_map={'':model_config['device']},local_files_only=True)
    model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,
        target_modules=config['target_modules'],bias='none'))
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.train()
    batch={k:torch.tensor([sample[k]],device=model_config['device']) for k in ('input_ids','labels')}
    batch['attention_mask']=torch.ones_like(batch['input_ids'])
    head_gradients = []
    def capture_input(module, inputs):
        inputs[0].register_hook(lambda gradient: head_gradients.append(gradient.cpu().clone()))
    hook = model.get_base_model().lm_head.register_forward_pre_hook(capture_input)
    torch.cuda.reset_peak_memory_stats()
    loss=model(**batch).loss
    loss.backward()
    parameters=[p for p in model.parameters() if p.requires_grad]
    old=[p.grad.cpu().clone() for p in parameters]
    original_memory=torch.cuda.max_memory_allocated()/2**20
    original_loss=loss.item()
    del loss
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    repeated_loss = model(**batch).loss
    repeated_loss.backward()
    repeated = [p.grad.cpu().clone() for p in parameters]
    repeat_numerator = sum((a-b).square().sum().item() for a,b in zip(old,repeated))
    repeat_denominator = sum(a.square().sum().item() for a in old)
    repeat_relative_l2 = (repeat_numerator/max(repeat_denominator,1e-30))**.5
    del repeated_loss
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    new_loss=assistant_loss(model,batch)
    new_loss.backward()
    new=[p.grad.cpu() for p in parameters]
    max_diff=max((a-b).abs().max().item() for a,b in zip(old,new))
    numerator=sum((a-b).square().sum().item() for a,b in zip(old,new))
    denominator=sum(a.square().sum().item() for a in old)
    relative_l2=(numerator/max(denominator,1e-30))**.5
    # BF16 GEMM shape changes can change rounding; FP32 algebra is unit-tested.
    # Compare the upstream gradient at the unchanged decoder boundary exactly.
    # Full-vs-full repetition measures reference-kernel nondeterminism separately.
    boundary_equal = torch.equal(head_gradients[0], head_gradients[2])
    hook.remove()
    passed=original_loss==new_loss.item() and boundary_equal
    result={'passed':passed,'tokens':len(sample['input_ids']), 'task_id':sample['task_id'],
        'original_loss':original_loss,'selective_loss':new_loss.item(),
        'gradient_max_abs_diff':max_diff,'gradient_relative_l2':relative_l2,
        'decoder_boundary_gradient_equal':boundary_equal,
        'full_repeat_gradient_relative_l2':repeat_relative_l2,
        'full_repeat_boundary_equal':torch.equal(head_gradients[0],head_gradients[1]),
        'criterion':'exact loss and exact decoder-boundary gradient; FP32 algebra tested separately',
        'original_peak_mib':original_memory,'selective_peak_mib':torch.cuda.max_memory_allocated()/2**20}
    path=Path('artifacts/m2_loss_equivalence.json')
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
    if not passed: raise RuntimeError('loss or gradient equivalence check failed')


if __name__=='__main__':
    main()
