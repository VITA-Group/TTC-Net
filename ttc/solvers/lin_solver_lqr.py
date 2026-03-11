import torch

# Batched matrix-vector multiply.
def bmv(mat, vec):
    """Performs batched matrix-vector multiplication."""
    return (mat @ vec.unsqueeze(-1)).squeeze(-1)

# Batched quadratic form: v^T M v.
def bvmv(vec, mat):
    """Performs batched vector-matrix-vector multiplication."""
    return (vec.unsqueeze(-2) @ mat @ vec.unsqueeze(-1)).squeeze(-1).squeeze(-1)

def solve_traj_states(u_traj, A_diag, B, Q, R_diag, h, r_traj=None, return_states=False, return_costates=False, return_lambda0=False):
    """
    Computes the total quadratic cost for a given control trajectory with time-varying dynamics/costs.
    
    Assumes inputs are time-varying:
        A, B, Q, R, r are shaped [bs, T, ...].
        
    Args:
        u_traj (torch.Tensor): Control trajectory. Shape: [bs, T, d].
        A_diag (torch.Tensor): Batched time-varying A (diagonal). Shape: [bs, T, d].
        B (torch.Tensor): Batched time-varying B. Shape: [bs, T, d, d].
        Q (torch.Tensor): Batched time-varying Q. Shape: [bs, T, d, d].
        R_diag (torch.Tensor): Batched time-varying R (diagonal). Shape: [bs, T, d].
        h (torch.Tensor): Batched initial state h_0. Shape: [bs, d].
        T (int): Time horizon.
        r_traj (torch.Tensor, optional): Affine cost on u. Shape: [bs, T, d].

    Returns:
        torch.Tensor or tuple: Total cost and optional states/costates.
    """
    bs, T, d = A_diag.shape
    du = B.shape[-1]
    device = A_diag.device
    
    # Normalize optional inputs.
    if r_traj is None:
        r_traj = torch.zeros(bs, T, du, device=device)
    if h is None:
        h = torch.zeros(bs, d, device=device)

    # Expand diagonal forms for batched matmul.
    A = torch.diag_embed(A_diag)
    R = torch.diag_embed(R_diag)
    
    if return_states:
        states = torch.zeros(bs, T, d, device=device)  # Stores h_1..h_T
        
    total_cost = torch.zeros(bs, device=device)
    x_t = h

    # Forward rollout and running cost accumulation.
    for t in range(T):
        u_t = u_traj[:, t, :]
        
        A_t = A[:, t]
        B_t = B[:, t]
        Q_t = Q[:, t]
        R_t = R[:, t]
        r_t = r_traj[:, t]

        # State transition: h_t -> h_{t+1}
        x_t = bmv(A_t, x_t) + bmv(B_t, u_t)

        # Running quadratic cost.
        running_cost = 0.5 * bvmv(x_t, Q_t) + 0.5 * bvmv(u_t, R_t) + torch.sum(u_t * r_t, dim=-1)
        total_cost += running_cost
        
        if return_states:
            states[:, t, :] = x_t

    return_list = [total_cost]
    if return_states:
        return_list.append(states)

    # Optional backward pass for costates.
    if return_costates:
        # costates[:, t] corresponds to lambda_{t+1}
        costates = torch.zeros(bs, T, d, device=device)
        
        # Terminal condition: lambda_T = Q_{T-1} h_T
        Q_last = Q[:, T-1]
        h_last = states[:, T-1]
        costates[:, T-1, :] = bmv(Q_last, h_last)

        # Backward recurrence: lambda_{t+1} = Q_t h_{t+1} + A_{t+1}^T lambda_{t+2}
        for t in range(T-2, -1, -1):
            Q_t = Q[:, t]
            h_next = states[:, t]           # h_{t+1}
            lambda_next_next = costates[:, t+1] # lambda_{t+2}
            
            A_next = A[:, t+1]
            
            val = bmv(Q_t, h_next) + bmv(A_next.transpose(-1, -2), lambda_next_next)
            costates[:, t, :] = val
            
        return_list.append(costates)

        if return_lambda0:
            # lambda_0 = A_0^T lambda_1
            lambda1 = costates[:, 0, :]
            A_0 = A[:, 0]
            lambda0 = bmv(A_0.transpose(-1, -2), lambda1)
            return_list.append(lambda0)

    if len(return_list) == 1:
        return return_list[0]
    else:
        return tuple(return_list)


def solve_traj_actions(A_diag, B, Q, R_diag, h, r_traj=None, return_traj=False):
    """
    Time-varying version of your homogeneous lin_solve, returning u_0 only.
    Matches lin_solve exactly when inputs are time-invariant and r_traj=None.

    Shapes:
      A_diag: [bs,T,d] (diag) or [bs,T,d,d]
      B:      [bs,T,d,d]
      Q:      [bs,T,d,d]
      R_diag: [bs,T,d]
      h:      [bs,d]
      r_traj: [bs,T,d] or None
    """
    bs, T, ns, nc = B.shape
    device = h.device
    dtype = Q.dtype
    n_sc = ns + nc

    A = torch.diag_embed(A_diag)
    R = torch.diag_embed(R_diag)

    # Augmented dynamics F_t = [A_t, B_t]
    F = torch.cat([A, B], dim=-1)  # [bs,T,ns,ns+nc]

    # Block quadratic cost [[Q_t, 0], [0, R_t]].
    Q_cost = torch.zeros(bs, T, n_sc, n_sc, device=device, dtype=dtype)
    Q_cost[:, 1:, :ns, :ns] = Q[:, :-1]
    Q_cost[:, :, ns:, ns:] = R

    # Linear term p_t only affects control coordinates.
    p_cost = torch.zeros(bs, T, n_sc, device=device, dtype=dtype)
    if r_traj is not None:
        p_cost[:, :, ns:] = r_traj

    # Terminal value initialization.
    V = Q[:, -1]
    v = torch.zeros(bs, ns, device=device, dtype=dtype)

    # Feedback/feedforward gains per timestep.
    K_gains = torch.zeros(bs, T, nc, ns, device=device, dtype=dtype)
    k_gains = torch.zeros(bs, T, nc, device=device, dtype=dtype)

    for t in range(T - 1, -1, -1):
        V_next, v_next = V, v

        Qt_t = Q_cost[:, t]      # [bs,ns+nc,ns+nc]
        pt_t = p_cost[:, t]      # [bs,ns+nc]
        F_t  = F[:, t]           # [bs,ns,ns+nc]

        FtT = F_t.transpose(-1, -2)  # [bs,ns+nc,ns]

        Qt_aug = Qt_t + FtT @ V_next @ F_t
        qt_aug = pt_t + bmv(FtT, v_next)

        Qxx = Qt_aug[:, :ns, :ns]
        Qxu = Qt_aug[:, :ns, ns:]
        Qux = Qt_aug[:, ns:, :ns]
        Quu = Qt_aug[:, ns:, ns:]

        qx = qt_aug[:, :ns]
        qu = qt_aug[:, ns:]

        Quu32 = Quu.to(torch.float32)
        Kt = -torch.linalg.solve(Quu32, Qux.to(torch.float32)).to(dtype)
        kt = -torch.linalg.solve(Quu32, qu.unsqueeze(-1).to(torch.float32)).squeeze(-1).to(dtype)

        K_gains[:, t] = Kt
        k_gains[:, t] = kt

        # Value recursion.
        KtT = Kt.transpose(-1, -2)
        V = Qxx + Qxu @ Kt + KtT @ Qux + KtT @ Quu @ Kt
        v = qx + bmv(Qxu, kt) + bmv(KtT, qu) + bmv(KtT @ Quu, kt)

    if return_traj:
        u_traj = torch.zeros(bs, T, nc, device=device, dtype=dtype)
        x_t = h # Set initial state
        
        for t in range(T):
            Kt = K_gains[:, t]
            kt = k_gains[:, t]
            
            # Optimal control.
            u_t = bmv(Kt, x_t) + kt
            u_traj[:, t] = u_t
            
            # State rollout.
            if t < T - 1:
                x_t = bmv(A[:, t], x_t) + bmv(B[:, t], u_t)

        return u_traj

    else:
        K0 = K_gains[:, 0]
        k0 = k_gains[:, 0]

        # First-step optimal action.
        u_0 = bmv(K0, h) + k0

        return u_0
