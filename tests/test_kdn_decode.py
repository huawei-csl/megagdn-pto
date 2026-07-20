from ref_kda_decode import fused_recurrent_kda
import torch
import torch.nn.functional as F 



# A simple example on how to run the KDA decode in megagdn-pto/tests/ref_kda_decody.py
# TODO: add 

def main():
    B, T, H, D = 1, 50, 96, 128

    print(f"Testing shape: [{B}, {T}, {H}, {D}]")
    torch.manual_seed(0)

    q = F.normalize(torch.randn((B, T, H, D), dtype=torch.float32), p=2, dim=-1).to(torch.bfloat16).npu()
    k = F.normalize(torch.randn((B, T, H, D), dtype=torch.float32), p=2, dim=-1).to(torch.bfloat16).npu()
    v = torch.randn((B, T, H, D), dtype=torch.bfloat16).npu()
    g = torch.randn((B, T, H, D), dtype=torch.bfloat16).npu()
    beta = torch.randn((B, T, H), dtype=torch.bfloat16).npu()



    # reference
    out, final_state = fused_recurrent_kda(q,k,v,g,beta)

    # call our kernel and see if they match...


main()