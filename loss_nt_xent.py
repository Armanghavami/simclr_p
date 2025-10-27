import torch.nn.functional as F 
import torch 




def nl_xent(z_i,z_j,temp : 0.5):

    N = z_i.shape[0]

    # adding the two clusters of augmentation and then normlizing it 
    z = torch.cat([z_i,z_j],dim=0)
    z = F.normalize(z,dim=1) # (2N,d)

    # calculating the cos similarity (2N,d)*(d,2N)=(2N,2N)

    sim = torch.matmul(z,z.T)

    # make a matrix with 1 and 0 and the 1 reprosent the similarity of the two 
    # identical points which should be excluded at the end for loss calculation 

    mask = torch.eye(n=2*N,dtype=bool, device=z.device)

    #  set the identical similarities to a small negative number 
    sim = sim.masked_fill(mask,-1e9)


    # this part is for calculating the positives in both ways (zi,zj) and then (zj,zi)
    pos_indices = torch.arange(N, device=z.device)
    positives = torch.cat([
    sim[pos_indices, pos_indices + N],
    sim[pos_indices + N, pos_indices]], dim=0)



    sim = sim / temp
    positives = positives / temp

    # suming and making the formula from the sim matrix 
    numerator = torch.exp(positives)
    denominator = torch.exp(sim).sum(dim=1)

    # loss and making the mean of loss from the loos batch 
    loss = -torch.log(numerator / denominator)
    return loss.mean()