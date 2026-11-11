import tilelang, tilelang.language as T, torch, math
# DeepSeek MLA (simplified, no RoPE decomposition): 
# cache is latent c_KV (Dc dim) + shared k_R. Q is (HQ, Dq). 
# absorbed form: Q -> (HQ, Dc) via Q*W_UK^T, then scores = Qabs @ c_KV^T ; V = c_KV @ W_UV
# Here test the ABSORBED decode: cache stores only c_KV (Dc=512) not full K,V (Dtens*2)
import torch
torch.manual_seed(0)
HQ=128; Dc=512; DK=128; DV=128; QLEN=16; KVLEN=8192
# absorbed: Qa (HQ,QLEN,Dc) ; KV cache c (KVLEN,Dc); W_UV (Dc,DV)
Qa=torch.randn(HQ,QLEN,Dc,device="cuda",dtype=torch.float16)*0.1
cKV=torch.randn(KVLEN,Dc,device="cuda",dtype=torch.float16)
Wuv=torch.randn(Dc,DV,device="cuda",dtype=torch.float16)*0.1
# reference: full attention in absorbed space
# scores = Qa @ cKV^T ; P=softmax ; out = P @ cKV @ Wuv
S=torch.einsum('hqd,kd->hqk',Qa.float(),cKV.float())/math.sqrt(Dc)
P=torch.softmax(S,dim=-1)
Oref=torch.einsum('hqk,kd->hqd',P,cKV.float())@Wuv.float()
print("ref done", Oref.shape, Oref.abs().mean().item())
# bytes: absorb-cache = QLEN*Dc + KVLEN*Dc (single latent!) + Wuv ; vs MHA would be KVLEN*DK*2
mla_bytes=HQ*QLEN*Dc*2 + KVLEN*Dc*2
mha_bytes=HQ*QLEN*DK*2 + HQ*KVLEN*DK*2*2
print(f"MLA cache bytes={mla_bytes/1e6:.1f}MB  vs MHA(128head)={mha_bytes/1e6:.1f}MB  ratio={mla_bytes/mha_bytes:.3f}")
FLOP_tot=2*HQ*QLEN*KVLEN*Dc + HQ*QLEN*KVLEN*Dc*2 + HQ*QLEN*Dc*DV*2
print(f"MLA absorbed FLOP={FLOP_tot/1e9:.1f}G int={FLOP_tot/(mla_bytes+KV_len:=0 and 0)}")