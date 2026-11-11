import torch
torch.manual_seed(0)
M,K=16,2048; BN=128; NT=392; N=NT*BN
X=torch.randn(M,K,device="cuda",dtype=torch.float16)*0.3
# realistic-ish: token logit = freq_bias[token] + X·w[token], freq_bias makes high-freq tokens consistently win
# vocab sorted by frequency -> top tokens cluster in the first tiles
def gen(freq_fn,name):
    torch.manual_seed(3)
    W=torch.randn(N,K,device="cuda",dtype=torch.float16)*0.05
    bias=freq_fn(torch.arange(N,device="cuda")).float()
    logits=X.float()@W.float().T + bias.unsqueeze(0)
    Wt=W.float().view(NT,BN,K); G=Wt.mean(1)
    # coarse screen including a per-tile bias (tile mean of bias) -- realistic: we know tile "popularity"
    Ehbias=bias.view(NT,BN).mean(1)
    coarse=X.float()@G.T + Ehbias.unsqueeze(0)
    true1=logits.argmax(1); true10=logits.topk(10,dim=1).indices
    for kept in (0.25,0.10,0.05):
        ntile=int(NT*kept); top=coarse.topk(ntile,dim=1).indices
        hit1=sum((top[b]==(true1[b]//BN)).any().item() for b in range(M))
        cnt=0
        for b in range(M):
            sel=set(top[b].tolist())
            for t in true10[b].tolist():
                if (t//BN) in sel: cnt+=1
        print(f"  {name} kept={kept:.2f}: top1={hit1}/{M} top10-recall={cnt/(M*10):.2f}")
# zipf-like: bias high for low index
gen(lambda i: -torch.log(i.float()+1.0),"zipf-sorted (freq order)")
# no bias (pure random) as control
gen(lambda i: torch.zeros_like(i.float()),"no freq bias")