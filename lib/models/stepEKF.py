import torch

def stepEKF(W, P, H, e, Q, R):

    H_T = torch.transpose(H, 0, -1) 
    A = ( R + H @ P @ H_T)

    K = P @ torch.linalg.solve(A, H_T, left=False)

    if len(e.size()) == 1:
        W = W + K * e
    else:
        W = W + K @ e

    P = P - K @ H @ P + Q

    return W, P