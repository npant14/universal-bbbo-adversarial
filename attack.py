import torch
import numpy

def hotflip(averaged_gradient, embedding_matrix, adv_token_ids, num_candidates=1):
    embedding_adv = torch.nn.functional.embedding(torch.IntTensor(adv_token_ids), embedding_matrix).detach()

    grad_embedding = torch.einsum("ij,kj->ik", (averaged_gradient, embedding_matrix))
    # this is assuming average gradient is squeezed and 2 dims
    # can consider adding dim 0 if makes computation simpler elsewhere
    if num_candidates > 1:
        elts, best_k_ids = torch.topk(grad_embeddings, num_candidates, dim=1)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_step = grad_embedding.max(1)
    return best_at_step[0].detach().cpu().numpy()
    
