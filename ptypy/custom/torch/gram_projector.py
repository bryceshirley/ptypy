import numpy as np
from scipy import ndimage

def gram_null_projectors(probe, psize, lam, z_list, qy, qx, band_idx, eig_tol=0.05, shear_sign=1.0):
    """
    Computes per-frequency null projectors of the ISS block Gram matrix.
    Uses fast map_coordinates interpolation for the parallax shear.
    """
    N = len(z_list)
    
    # 1. Mode-summed autocorrelation on the probe grid, normalized to C(0)=1
    A = np.zeros(probe.shape[-2:])
    for m in range(probe.shape[0]):
        A += np.abs(np.fft.fft2(probe[m])) ** 2
    C = np.fft.fftshift(np.fft.ifft2(A))
    cy, cx = C.shape[0] // 2, C.shape[1] // 2
    C = C / C[cy, cx]
    
    qyb, qxb = qy[band_idx], qx[band_idx]
    q2 = qyb ** 2 + qxb ** 2
    npix = len(qyb)
    
    # 2. Build the Block Gram Matrix
    M = np.empty((npix, N, N), dtype=np.complex128)
    for j in range(N):
        for k in range(N):
            dz = z_list[j] - z_list[k]
            if dz == 0:
                M[:, j, k] = 1.0
                continue
            
            # Parallax shear in probe-grid pixels
            sy = shear_sign * lam * dz * qyb / psize + cy
            sx = shear_sign * lam * dz * qxb / psize + cx
            
            # Fast sub-pixel interpolation
            cre = ndimage.map_coordinates(C.real, [sy, sx], order=1, cval=0.0)
            cim = ndimage.map_coordinates(C.imag, [sy, sx], order=1, cval=0.0)
            
            # Block Gram element: Fresnel phase * Parallax shear
            M[:, j, k] = np.exp(1j * np.pi * lam * dz * q2) * (cre + 1j * cim)
            
    # Hermitize to correct any floating point asymmetry
    M = 0.5 * (M + np.conj(np.transpose(M, (0, 2, 1))))
    
    # 3. Eigendecomposition to isolate the null space
    evals, evecs = np.linalg.eigh(M)
    null = evals < eig_tol * evals[:, -1:]
    
    U = np.where(null[:, None, :], evecs, 0.0)
    P_null = U @ np.conj(np.transpose(U, (0, 2, 1)))
    
    return P_null.astype(np.complex64), null.sum(axis=1)


def apply_gram_pocs(O, P_null, band_idx, supports, n_pocs=30):
    """
    Data-neutral null-space excision. 
    Projects between the determined data subspace and an externally provided free-space prior.
    """
    N = O.shape[0]
    g = (O - 1.0).astype(np.complex64)
    
    # Extract the data-determined component (fixed by the physics)
    G0 = np.fft.fft2(g, axes=(-2, -1))
    v0 = G0.reshape(N, -1)[:, band_idx[0]].T
    det0 = v0 - np.einsum('pjk,pk->pj', P_null, v0) 
    
    gk = g.copy()
    for _ in range(n_pocs):
        # 1. Project onto the free-space prior (using external support masks)
        gk = gk * supports  
        
        # 2. Project back onto the data-consistent subspace
        Gk = np.fft.fft2(gk, axes=(-2, -1))
        vk = Gk.reshape(N, -1)[:, band_idx[0]].T
        vk = det0 + np.einsum('pjk,pk->pj', P_null, vk)
        
        # 3. Restore exact out-of-band high frequencies
        Gk = G0.copy()
        Gk.reshape(N, -1)[:, band_idx[0]] = vk.T
        gk = np.fft.ifft2(Gk, axes=(-2, -1)).astype(np.complex64)
        
    return 1.0 + gk