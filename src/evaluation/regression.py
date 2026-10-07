"""Frozen general-tool and synthetic page regression, separate from real-report quality."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel
import torch
from transformers import AutoProcessor, AutoTokenizer
import yaml

from src.evaluation.m0_inference_check import generate, load_model, meets_expectation, parse_tool_calls


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--adapter',type=Path)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    config=yaml.safe_load(Path('configs/m0_inference.yaml').read_text())
    model_path=config['model']['path']
    tasks=list(map(json.loads,Path('configs/m0_smoke_tasks.jsonl').read_text().splitlines()))
    tools=json.loads(Path('configs/m0_smoke_tools.json').read_text())
    schemas={t['function']['name']:t['function']['parameters'] for t in tools}
    visual=json.loads(Path('configs/m2_visual_regression.json').read_text())
    torch.manual_seed(0)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=load_model(model_path,None,config['model']['device'])
    if args.adapter:
        model=PeftModel.from_pretrained(model,args.adapter)
        model.eval()
    records=[]
    for task in tasks:
        row=generate(model,tokenizer,task['messages'],tools if task['use_tools'] else None,
                     config['generation'],512,4096,config['model']['device'])
        calls, errors=parse_tool_calls(row.get('text',''),schemas)
        row.update(id=task['id'],modality='text',passed=('error' not in row and not errors
            and row['stopped_on_eos'] and meets_expectation(task['expect'],row['text'],calls)))
        records.append(row)
    for task in visual:
        image=Image.new('RGB',(700,240),'white')
        draw=ImageDraw.Draw(image)
        font=ImageFont.load_default(size=28)
        for i,line in enumerate(task['lines']):
            draw.text((15,15+i*45),line,fill='black',font=font)
        image.save(args.output_dir/f"{task['id']}.png")
        messages=[{'role':'user','content':[{'type':'image','image':image},
                    {'type':'text','text':task['question']}]}]
        encoded=processor.apply_chat_template(messages,add_generation_prompt=True,
            enable_thinking=False,tokenize=True,return_dict=True,return_tensors='pt')
        encoded=encoded.to(config['model']['device'])
        length=encoded['input_ids'].shape[1]
        started=time.monotonic()
        with torch.inference_mode():
            output=model.generate(**encoded,max_new_tokens=128,do_sample=False,
                eos_token_id=[tokenizer.eos_token_id,tokenizer.pad_token_id],
                pad_token_id=tokenizer.pad_token_id)
        text=tokenizer.decode(output[0,length:],skip_special_tokens=True).strip()
        records.append({'id':task['id'],'modality':'synthetic_image','text':text,
            'passed':text==task['answer'],'prompt_tokens':length,
            'image_grid_thw':encoded['image_grid_thw'].tolist(),
            'latency_s':time.monotonic()-started})
    (args.output_dir/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    paths=['configs/m0_smoke_tasks.jsonl','configs/m0_smoke_tools.json','configs/m2_visual_regression.json']
    summary={'adapter':str(args.adapter) if args.adapter else None,
        'passed':{m:sum(r['passed'] for r in records if r['modality']==m)
                  for m in ('text','synthetic_image')},
        'totals':{'text':len(tasks),'synthetic_image':len(visual)},
        'frozen_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        'config':config,'limitations':'Synthetic visual regression only; not TAT-DQA or report acceptance.'}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary['passed']))


if __name__=='__main__':
    main()
