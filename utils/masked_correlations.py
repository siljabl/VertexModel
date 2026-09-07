import numpy as np
from tqdm  import tqdm
from numba import njit, prange
import warnings


############################
# Temporal correlations    #
############################

def scalar_temporal_correlation(var1, var2, Nframes, t_max=None):
    """
    Temporal correlation of two scalar fields.

    Parameters
    ----------
    var1, var2 : masked arrays, shape (Nframes, Npoints)
    Nframes : int
    t_max : int or None

    Returns
    -------
    C_norm : masked array, shape (Nframes, t_max)
        Normalized correlation C(i, j - i).
    N      : masked array, shape (Nframes, t_max)
        Number of points used in each correlation.
    delta_f : masked array, shape (Nframes, t_max)
        Frame lag j - i.
    """
    if t_max is None:
        t_max = Nframes

    delta_f = np.ma.masked_array(np.zeros((Nframes, Nframes)), True)
    C_norm  = np.ma.masked_array(np.zeros((Nframes, Nframes)), True)
    N       = np.ma.masked_array(np.zeros((Nframes, Nframes)), True)

    for i in tqdm(range(Nframes)):
        if np.any(var1[i, :].mask == False):
            Noki = len(var1[i, :].compressed())
            if Noki > 0:
                bool_oki = ~var1.mask[i, :]

                for j in range(i, Nframes):
                    if np.any(var1[j, :].mask == False):
                        Nokj = len(var1[j, :].compressed())
                        if Nokj > 0:
                            bool_okj = ~var1.mask[j, :]
                            bool_ok  = bool_oki * bool_okj

                            var1_i = var1[i, bool_ok].compressed()
                            var1_j = var1[j, bool_ok].compressed()
                            var2_i = var2[i, bool_ok].compressed()
                            var2_j = var2[j, bool_ok].compressed()

                            rms = np.ma.sqrt(
                                abs(np.ma.mean(var1_i * var2_i) *
                                    np.ma.mean(var1_j * var2_j))
                            )
                            C_norm[i, j - i] = np.ma.mean(var1_i * var2_j) / rms
                            N[i, j]          = np.min([Noki, Nokj])
                            delta_f[i, j]    = j - i

    return C_norm[:, :t_max], N[:, :t_max], delta_f[:, :t_max]


def scalar_vector_temporal_correlation(var1, vec2, Nframes, t_max=None):
    """
    Temporal correlation between a scalar and a vector field.
    """
    if t_max is None:
        t_max = Nframes

    var2x, var2y = vec2

    delta_f = np.ma.masked_array(np.zeros((Nframes, Nframes)), True)
    C_norm  = np.ma.masked_array(np.zeros((Nframes, Nframes)), True)
    N       = np.ma.masked_array(np.zeros((Nframes, Nframes)), True)

    for i in tqdm(range(Nframes)):
        if np.any(var2x[i, :].mask == False):
            Noki = len(var2x[i, :].compressed())
            if Noki > 0:
                bool_oki = ~var2x.mask[i, :] * ~var2y.mask[i, :]

                for j in range(i, Nframes):
                    if np.any(var2x[j, :].mask == False):
                        Nokj = len(var2x[j, :].compressed())
                        if Nokj > 0:
                            bool_okj = ~var2x.mask[j, :] * ~var2y.mask[j, :]
                            bool_ok  = bool_oki * bool_okj

                            var1_i = var1[i, bool_ok].compressed()
                            var1_j = var1[j, bool_ok].compressed()

                            var2x_i = var2x[i, bool_ok].compressed()
                            var2x_j = var2x[j, bool_ok].compressed()
                            var2y_i = var2y[i, bool_ok].compressed()
                            var2y_j = var2y[j, bool_ok].compressed()

                            num = np.ma.mean(
                                var1_i * (var2x_j + var2y_j)
                            )
                            den = np.ma.sqrt(
                                abs(
                                    np.ma.mean(var1_i * (var2x_i + var2y_i)) *
                                    np.ma.mean(var1_j * (var2x_j + var2y_j))
                                )
                            )
                            C_norm[i, j - i] = num / den
                            N[i, j]          = np.min([Noki, Nokj])
                            delta_f[i, j]    = j - i

    return C_norm[:, :t_max], N[:, :t_max], delta_f[:, :t_max]


def vector_temporal_correlation(vec1, vec2, Nframes, t_max=None):
    """
    Temporal correlation between two vector fields.
    """
    if t_max is None:
        t_max = Nframes

    var1x, var1y = vec1
    var2x, var2y = vec2

    delta_f = np.ma.masked_array(np.zeros((Nframes, Nframes)), True)
    C_norm  = np.ma.masked_array(np.zeros((Nframes, Nframes)), True)
    N       = np.ma.masked_array(np.zeros((Nframes, Nframes)), True)

    for i in tqdm(range(Nframes)):
        if np.any(var1x[i, :].mask == False):
            Noki = len(var1x[i, :].compressed())
            if Noki > 0:
                bool_oki = ~var1x.mask[i, :] * ~var1y.mask[i, :]

                for j in range(i, Nframes):
                    if np.any(var1x[j, :].mask == False):
                        Nokj = len(var1x[j, :].compressed())
                        if Nokj > 0:
                            bool_okj = ~var1x.mask[j, :] * ~var1y.mask[j, :]
                            bool_ok  = bool_oki * bool_okj

                            var1x_i = var1x[i, bool_ok].compressed()
                            var1x_j = var1x[j, bool_ok].compressed()
                            var1y_i = var1y[i, bool_ok].compressed()
                            var1y_j = var1y[j, bool_ok].compressed()

                            var2x_i = var2x[i, bool_ok].compressed()
                            var2x_j = var2x[j, bool_ok].compressed()
                            var2y_i = var2y[i, bool_ok].compressed()
                            var2y_j = var2y[j, bool_ok].compressed()

                            num = np.ma.mean(
                                var1x_i * var2x_j + var1y_i * var2y_j
                            )
                            den = np.ma.sqrt(
                                abs(
                                    np.ma.mean(var1x_i * var2x_i +
                                               var1y_i * var2y_i) *
                                    np.ma.mean(var1x_j * var2x_j +
                                               var1y_j * var2y_j)
                                )
                            )
                            C_norm[i, j - i] = num / den
                            N[i, j]          = np.min([Noki, Nokj])
                            delta_f[i, j]    = j - i

    return C_norm[:, :t_max], N[:, :t_max], delta_f[:, :t_max]


def general_temporal_correlation(var1, var2=None, t_max=None, t_avrg=False):
    """
    Wrapper for temporal correlations, handling scalar / vector combinations.
    """
    if var2 is None:
        var2 = var1

    dim_var1 = np.shape(var1)
    dim_var2 = np.shape(var2)

    assert len(dim_var1) in (2, 3) and len(dim_var2) in (2, 3)

    if len(dim_var1) == 2:
        Nframes = dim_var1[0]

        if len(dim_var2) == 2:
            C_norm, N, delta_f = scalar_temporal_correlation(
                var1, var2, Nframes, t_max
            )
        else:
            var2x, var2y = var2
            C_norm, N, delta_f = scalar_vector_temporal_correlation(
                var1, [var2x, var2y], Nframes, t_max
            )

    else:  # len(dim_var1) == 3
        Nframes = dim_var1[1]
        var1x, var1y = var1[0], var1[1]

        if len(dim_var2) == 2:
            C_norm, N, delta_f = scalar_vector_temporal_correlation(
                var2, [var1x, var1y], Nframes, t_max
            )
        else:
            var2x, var2y = var2
            C_norm, N, delta_f = vector_temporal_correlation(
                [var1x, var1y], [var2x, var2y], Nframes, t_max
            )

    if t_avrg:
        C_norm = np.mean(C_norm, axis=0)

    return {
        "delta_f": delta_f,
        "C_norm":  C_norm,
        "N":       N,
    }


########################
# Spatial correlations #
########################

@njit(parallel=True)
def scalar_spatial_correlation_loopv2(xf, yf, var1f, var2f, r_bin_edges):
    Nbins = len(r_bin_edges) - 1
    Nf_in_rbin_values = np.zeros(Nbins)
    Nf_in_rbin_mask   = np.ones(Nbins, dtype=np.bool_)

    Cf_values = np.zeros(Nbins)
    Cf_mask   = np.ones(Nbins, dtype=np.bool_)
    varf_rms  = np.sqrt(abs(np.mean(var1f * var2f)))

    for i in prange(Nbins):
        for j in range(len(var1f)):
            xf_i = xf[j]
            yf_i = yf[j]

            xf_rel = xf - xf_i
            yf_rel = yf - yf_i
            rf_rel = np.sqrt(xf_rel**2 + yf_rel**2)

            bool_in_bin = (rf_rel > r_bin_edges[i]) * (rf_rel <= r_bin_edges[i+1])

            if np.any(bool_in_bin):
                Cf_mask[i]          = False
                Nf_in_rbin_mask[i]  = False
                Nf_in_rbin_values[i] += np.sum(bool_in_bin)

                var2f_in_bin = var2f[bool_in_bin]
                var1var2     = var1f[j] * var2f_in_bin

                Cf_values[i] += np.sum(var1var2 / (varf_rms**2))

    return Nf_in_rbin_values, Nf_in_rbin_mask, Cf_values, Cf_mask


def scalar_spatial_correlation(x, y, var1, var2, dr, r_max, Nmax=5000,
                               every_n_frames=1):
    Nframes = x.shape[0]

    # Script 2 behavior: bins from 0, no special r=0 bin
    r_bin_edges   = np.arange(0, r_max, dr)
    r_bin_centers = (r_bin_edges[1:] + r_bin_edges[:-1]) / 2

    Nbins     = len(r_bin_centers)
    C_norm    = np.ma.masked_array(np.zeros((Nframes, Nbins)), True)
    N_in_rbin = np.ma.masked_array(np.zeros((Nframes, Nbins)), True)

    frame_axis        = np.arange(0, Nframes, every_n_frames, dtype=int)
    frame_axis_masked = np.ma.masked_array(np.arange(0, Nframes, dtype=int), True)

    for f in tqdm(frame_axis):
        if np.any(var1[f, :].mask == False):
            Nok = len(var1[f, :].compressed())
            if Nok > 0:
                xf  = x[f, :].compressed()
                yf  = y[f, :].compressed()
                v1f = var1[f, :].compressed()
                v2f = var2[f, :].compressed()

                if Nok >= Nmax:
                    ind_selected = np.random.choice(
                        np.arange(len(xf)), replace=False, size=Nmax
                    )
                    xf  = xf[ind_selected]
                    yf  = yf[ind_selected]
                    v1f = v1f[ind_selected]
                    v2f = v2f[ind_selected]

                frame_axis_masked.mask[f] = False

                Nvals, Nmask, Cvals, Cmask = scalar_spatial_correlation_loopv2(
                    xf, yf, v1f, v2f, r_bin_edges
                )

                C_norm[f, :]         = Cvals
                C_norm.mask[f, :]    = Cmask
                N_in_rbin[f, :]      = Nvals
                N_in_rbin.mask[f, :] = Nmask

    C_norm = C_norm / N_in_rbin

    return C_norm, N_in_rbin, r_bin_centers, frame_axis_masked


@njit(parallel=True)
def scalar_vector_spatial_correlation_loopv2(xf, yf, var1f, vec2f, r_bin_edges):
    var2xf, var2yf = vec2f

    Nbins = len(r_bin_edges) - 1
    Nf_in_rbin_values = np.zeros(Nbins)
    Nf_in_rbin_mask   = np.ones(Nbins, dtype=np.bool_)

    Cf_values = np.zeros(Nbins)
    Cf_mask   = np.ones(Nbins, dtype=np.bool_)
    varf_rms  = np.sqrt(np.mean(var1f * var2xf + var1f * var2yf))

    for i in prange(Nbins):
        for j in range(len(var2xf)):
            xf_i = xf[j]
            yf_i = yf[j]

            xf_rel = xf - xf_i
            yf_rel = yf - yf_i
            rf_rel = np.sqrt(xf_rel**2 + yf_rel**2)

            bool_in_bin = (rf_rel > r_bin_edges[i]) * (rf_rel <= r_bin_edges[i+1])

            if np.any(bool_in_bin):
                Cf_mask[i]          = False
                Nf_in_rbin_mask[i]  = False
                Nf_in_rbin_values[i] += np.sum(bool_in_bin)

                var1f_in_bin = var1f[bool_in_bin]
                var1var2     = var1f_in_bin * var2xf[j] + var1f_in_bin * var2yf[j]

                Cf_values[i] += np.sum(var1var2 / (varf_rms**2))

    return Nf_in_rbin_values, Nf_in_rbin_mask, Cf_values, Cf_mask


def scalar_vector_spatial_correlation(x, y, var1, vec2, dr, r_max, Nmax=5000,
                                      every_n_frames=1):
    var2x, var2y = vec2

    Nframes = x.shape[0]

    r_bin_edges   = np.arange(0, r_max, dr)
    r_bin_centers = (r_bin_edges[1:] + r_bin_edges[:-1]) / 2

    Nbins     = len(r_bin_centers)
    C_norm    = np.ma.masked_array(np.zeros((Nframes, Nbins)), True)
    N_in_rbin = np.ma.masked_array(np.zeros((Nframes, Nbins)), True)

    frame_axis        = np.arange(0, Nframes, every_n_frames, dtype=int)
    frame_axis_masked = np.ma.masked_array(np.arange(0, Nframes, dtype=int), True)

    for f in tqdm(frame_axis):
        if np.any(var2x[f, :].mask == False):
            Nok = len(var2x[f, :].compressed())
            if Nok > 0:
                xf  = x[f, :].compressed()
                yf  = y[f, :].compressed()
                v1f = var1[f, :].compressed()
                v2xf = var2x[f, :].compressed()
                v2yf = var2y[f, :].compressed()
                vec2f = (v2xf, v2yf)

                if Nok >= Nmax:
                    ind_selected = np.random.choice(
                        np.arange(len(xf)), replace=False, size=Nmax
                    )
                    xf   = xf[ind_selected]
                    yf   = yf[ind_selected]
                    v1f  = v1f[ind_selected]
                    v2xf = v2xf[ind_selected]
                    v2yf = v2yf[ind_selected]
                    vec2f = (v2xf, v2yf)

                frame_axis_masked.mask[f] = False

                Nvals, Nmask, Cvals, Cmask = scalar_vector_spatial_correlation_loopv2(
                    xf, yf, v1f, vec2f, r_bin_edges
                )

                C_norm[f, :]         = Cvals
                C_norm.mask[f, :]    = Cmask
                N_in_rbin[f, :]      = Nvals
                N_in_rbin.mask[f, :] = Nmask

    C_norm = C_norm / N_in_rbin

    return C_norm, N_in_rbin, r_bin_centers, frame_axis_masked


@njit(parallel=True)
def vector_spatial_correlation_loopv2(xf, yf, vec1f, vec2f, r_bin_edges):
    var1xf, var1yf = vec1f
    var2xf, var2yf = vec2f

    Nbins = len(r_bin_edges) - 1
    Nf_in_rbin_values = np.zeros(Nbins)
    Nf_in_rbin_mask   = np.ones(Nbins, dtype=np.bool_)

    Cvf_norm_values = np.zeros(Nbins)
    Cvf_mask        = np.ones(Nbins, dtype=np.bool_)
    vf_rms          = np.sqrt(np.mean(var1xf * var2xf + var1yf * var2yf))

    for i in prange(Nbins):
        for j in range(len(var1xf)):
            xf_i = xf[j]
            yf_i = yf[j]

            xf_rel = xf - xf_i
            yf_rel = yf - yf_i
            rf_rel = np.sqrt(xf_rel**2 + yf_rel**2)

            bool_in_bin = (rf_rel > r_bin_edges[i]) * (rf_rel <= r_bin_edges[i+1])

            if np.any(bool_in_bin):
                Cvf_mask[i]          = False
                Nf_in_rbin_mask[i]   = False
                Nf_in_rbin_values[i] += np.sum(bool_in_bin)

                var1xf_in_bin = var1xf[bool_in_bin]
                var1yf_in_bin = var1yf[bool_in_bin]

                vxvx_vyvy = var1xf_in_bin * var2xf[j] + var1yf_in_bin * var2yf[j]
                Cvf_norm_values[i] += np.sum(vxvx_vyvy / (vf_rms * vf_rms))

    return Nf_in_rbin_values, Nf_in_rbin_mask, Cvf_norm_values, Cvf_mask


def vector_spatial_correlation(x, y, vec1, vec2, dr, r_max):
    var1x, var1y = vec1
    var2x, var2y = vec2

    r_bin_edges   = np.arange(0, r_max, dr)
    r_bin_centers = (r_bin_edges[1:] + r_bin_edges[:-1]) / 2

    Nframes = x.shape[0]
    Nbins   = len(r_bin_centers)

    C_norm    = np.ma.masked_array(np.zeros((Nframes, Nbins)), True)
    N_in_rbin = np.ma.masked_array(np.zeros((Nframes, Nbins)), True)

    frame_axis        = np.arange(0, Nframes, 1, dtype=int)
    frame_axis_masked = np.ma.masked_array(np.arange(0, Nframes, dtype=int), True)

    for f in tqdm(frame_axis):
        if np.any(var1x[f, :].mask == False):
            Nok = len(var1x[f, :].compressed())
            if Nok > 0:
                xf    = x[f, :].compressed()
                yf    = y[f, :].compressed()
                v1xf  = var1x[f, :].compressed()
                v1yf  = var1y[f, :].compressed()
                v2xf  = var2x[f, :].compressed()
                v2yf  = var2y[f, :].compressed()
                vec1f = (v1xf, v1yf)
                vec2f = (v2xf, v2yf)

                frame_axis_masked.mask[f] = False

                Nvals, Nmask, Cvals, Cmask = vector_spatial_correlation_loopv2(
                    xf, yf, vec1f, vec2f, r_bin_edges
                )

                C_norm[f, :]         = Cvals
                C_norm.mask[f, :]    = Cmask
                N_in_rbin[f, :]      = Nvals
                N_in_rbin.mask[f, :] = Nmask

    C_norm = C_norm / N_in_rbin

    return C_norm, N_in_rbin, r_bin_centers, frame_axis_masked


def general_spatial_correlation(x, y, var1, var2=None, dr=40, r_max=500,
                                t_avrg=False):
    """
    Wrapper for spatial correlations, handling scalar / vector combinations.
    """
    if var2 is None:
        var2 = var1

    dim_var1 = np.shape(var1)
    dim_var2 = np.shape(var2)

    assert len(dim_var1) in (2, 3) and len(dim_var2) in (2, 3)

    if len(dim_var1) == 2:
        if len(dim_var2) == 2:
            C_norm, N_in_rbin, r_bin_centers, frame_axis_masked = \
                scalar_spatial_correlation(x, y, var1, var2, dr, r_max)
        else:
            var2x, var2y = var2
            C_norm, N_in_rbin, r_bin_centers, frame_axis_masked = \
                scalar_vector_spatial_correlation(x, y, var1, [var2x, var2y],
                                                  dr, r_max)

    else:  # len(dim_var1) == 3
        var1x, var1y = var1[0], var1[1]
        if len(dim_var2) == 2:
            C_norm, N_in_rbin, r_bin_centers, frame_axis_masked = \
                scalar_vector_spatial_correlation(x, y, var2, [var1x, var1y],
                                                  dr, r_max)
        else:
            var2x, var2y = var2
            C_norm, N_in_rbin, r_bin_centers, frame_axis_masked = \
                vector_spatial_correlation(x, y, [var1x, var1y],
                                           [var2x, var2y], dr, r_max)

    if t_avrg:
        C_norm       = np.mean(C_norm, axis=0)
        r_bin_centers = np.ma.array(r_bin_centers, mask=C_norm.mask)

    return {
        "C_norm":          C_norm,
        "N_pairs_in_rbin": N_in_rbin,
        "r_bin_centers":   r_bin_centers,
        "frame_axis":      frame_axis_masked,
    }
