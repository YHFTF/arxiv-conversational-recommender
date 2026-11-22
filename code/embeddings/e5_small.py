"""Pinned multilingual E5-small, mean pooling and role prefixes; CPU supported."""
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
MODEL='intfloat/multilingual-e5-small'
MANIFEST=ROOT/'output/e5_small/model.json'


class E5Small:
    def __init__(self,device='cpu',threads=4,output_dir=None):
        import json
        from transformers import AutoTokenizer,AutoModel
        output=Path(output_dir) if output_dir is not None else ROOT/'output'
        meta=json.loads((output/'e5_small/model.json').read_text(encoding='utf-8'));self.revision=meta['revision'];self.device=device
        torch.set_num_threads(threads)
        snapshot=output/'hf_cache/models--intfloat--multilingual-e5-small/snapshots'/self.revision
        args=dict(local_files_only=True,trust_remote_code=False)
        self.tokenizer=AutoTokenizer.from_pretrained(str(snapshot),**args)
        self.model=AutoModel.from_pretrained(str(snapshot),**args).float().to(device).eval()

    @torch.inference_mode()
    def encode(self,texts,query=False,batch_size=32):
        vectors=[];prefix='query: ' if query else 'passage: '
        for start in range(0,len(texts),batch_size):
            batch=self.tokenizer([prefix+t for t in texts[start:start+batch_size]],padding=True,truncation=True,max_length=512,return_tensors='pt').to(self.device)
            output=self.model(**batch).last_hidden_state
            mask=batch['attention_mask'].unsqueeze(-1)
            pooled=(output*mask).sum(1)/mask.sum(1).clamp(min=1)
            value=F.normalize(pooled.float(),dim=1).cpu()
            if not torch.isfinite(value).all():raise ValueError('Nonfinite E5 vector')
            vectors.append(value)
        return torch.cat(vectors) if vectors else torch.empty((0,384))


def download():
    import json
    from huggingface_hub import HfApi,snapshot_download
    info=HfApi().model_info(MODEL)
    names={s.rfilename for s in info.siblings}
    weight='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
    files=[weight,'config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model']
    snapshot_download(MODEL,revision=info.sha,cache_dir=str(ROOT/'output/hf_cache'),allow_patterns=[n for n in files if n in names])
    MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    MANIFEST.write_text(json.dumps(dict(model=MODEL,revision=info.sha,files=[n for n in files if n in names]),indent=2),encoding='utf-8')
    print('Pinned E5-small',info.sha,flush=True)


if __name__=='__main__':download()
