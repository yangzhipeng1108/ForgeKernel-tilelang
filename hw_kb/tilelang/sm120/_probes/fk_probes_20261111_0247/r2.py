import torch
torch.manual_seed(0)
M,K=16,2048; BN=128; NT=392; N=NT*BN  # 50176
def score_recall(W,struct):
    X=torch.randn(M,K,device="cuda",dtype=torch.float16)*0.3
    logits=(X.float()@W.float().T)
    Wt=W.float().view(NT,BN,K); G=Wt.mean(1)
    coarse=X.float()@G.T
    for kept in (0.25,0.10,0.05):
        ntile=int(NT*kept)
        top=coarse.topk(ntile,dim=1).indices
        true1=logits.argmax(1); true10=logits.topk(10,dim=1).indices
        hit1=sum((top[b]==(true1[b]//BN)).any().item() for b in range(M))
        cnt=0
        for b in range(M):
            sel=set(top[b].tolist())
            for t in true10[b].tolist():
                if (t//BN) in sel: cnt+=1
        print(f"  {struct} kept={kept:.2f} ntile={ntile}: top1 tile-recall={hit1}/{M}  top10 token-recall={cnt/(M*10):.2f}")
# tie-consistent comparison: X generated here is NOT consistent across calls (uses module X).
torch.manual_seed(1)
W_rand=torch.randn(N,K,device="cuda",dtype=torch.float16)*0.1
print("RANDOM W:")
score_recall(W_rand,"rand")
torch.manual_seed(2)
common=torch.randn(NT,K,device="cuda",dtype=torch.float16)*0.1
noise=torch.randn(NT,BN,K,device="cuda",dtype=torch.float16)*0.03
W_struct=(common.unsqueeze(1)+noise).reshape(N,K).contiguous()
print("STRUCTURED W (tile rows share a common component):")
score_recall(W_struct,"struct")