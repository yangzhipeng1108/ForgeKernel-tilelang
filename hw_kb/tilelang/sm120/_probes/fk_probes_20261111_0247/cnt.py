import math
# verify tile-skip is ~exactly half the k-block visits
for SEQ,BM,BN in [(4096,64,128),(4096,128,64),(4096,64,64),(2048,64,128),(8192,64,128)]:
    NQ=SEQ//BM
    total=0
    for bq in range(NQ):
        total+=math.ceil((bq+1)*BM/BN)
    full=NQ*(SEQ//BN)
    print(f"SEQ{SEQ} BM{BM} BN{BN}: skipvisits={total} fullvisits={full} ratio={total/full:.3f}")