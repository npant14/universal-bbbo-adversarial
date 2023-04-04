import torch
import numpy

def hotflip(averaged_gradient, embedding_matrix, adv_token_ids, num_candidates=1):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedding_adv = torch.nn.functional.embedding(torch.IntTensor(adv_token_ids).to(device), embedding_matrix).to(device).detach()

    grad_embedding = torch.einsum("ij,kj->ik", (averaged_gradient, embedding_matrix)).to(device)
    # this is assuming average gradient is squeezed and 2 dims
    # can consider adding dim 0 if makes computation simpler elsewhere
    if num_candidates > 1:
        elts, best_k_ids = torch.topk(grad_embeddings, num_candidates, dim=1)
        return best_k_ids.detach().to(device).numpy()[0]
    _, best_at_step = grad_embedding.max(1)
    return best_at_step[0].detach().to(device).cpu().numpy()
    
