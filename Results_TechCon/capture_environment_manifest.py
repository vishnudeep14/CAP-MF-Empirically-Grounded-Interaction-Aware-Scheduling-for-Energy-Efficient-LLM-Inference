from pathlib import Path
import json, platform, subprocess
import torch, transformers, vllm
from huggingface_hub import model_info
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
def cmd(args):
    p=subprocess.run(args,capture_output=True,text=True)
    return {"return_code":p.returncode,"stdout":p.stdout.strip(),"stderr":p.stderr.strip()}
info=model_info(MODEL)
manifest={
 "python_version":platform.python_version(),
 "pytorch_version":torch.__version__,
 "transformers_version":transformers.__version__,
 "vllm_version":vllm.__version__,
 "cuda_runtime_from_pytorch":torch.version.cuda,
 "gpu_names":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
 "nvidia_smi_query":cmd(["nvidia-smi","--query-gpu=index,name,driver_version,memory.total,compute_cap","--format=csv,noheader"]),
 "model":MODEL,
 "model_revision_sha":info.sha,
 "model_last_modified":str(info.last_modified) if info.last_modified else None,
}
Path("techcon_environment_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(json.dumps(manifest,indent=2))
