import numpy as np
from scipy.linalg import hadamard

# =====================================================================
#
# ---------------------- Vectorization of Sboxes ----------------------
#
# =====================================================================

# Prince Sbox table
prince_sbox = np.array([0xb, 0xf, 0x3, 0x2, 0xa, 0xc, 0x9, 0x1, 0x6, 0x7, 0x8, 0x0, 0xe, 0x5, 0xd, 0x4])

def output_prince_sbox(pt, key):
    return prince_sbox[pt ^ key]

# Vectorization of Prince Sbox output computation
prince_sbox_vectorized = np.vectorize(output_prince_sbox)

# Clyde128 Sbox table
sbox_clyde = np.array([0x0, 0x8, 0x1, 0xF, 0x2, 0xA, 0x7, 0x9, 0x4, 0xD, 0x5, 0x6, 0xE, 0x3, 0xB, 0xC])

def output_clyde_sbox(pt, key):
    return sbox_clyde[pt ^ key]

# Vectorization of Clyde128 Sbox output computation
clyde_sbox_vectorized = np.vectorize(output_clyde_sbox)

# AES Sbox table
AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])

def output_aes_sbox(pt, key):
    return AES_Sbox[pt ^ key]

# Vectorization of AES Sbox output computation
aes_sbox_vectorized = np.vectorize(output_aes_sbox)



# =====================================================================
#
# --------------------- Computation of WH basis -----------------------
#
# =====================================================================

def orthonormal_basis_projection(
    targeted_values, max_nb_monomials_interactions, len_basis
):
    """
    Implementation of the orthonormal basis proposed by Guilley etal. in [GHMR17, Theorem 5].

    [GHMR17]  Sylvain Guilley, Annelie Heuser, Tang Ming, and Olivier Rioul. Stochastic side-channel leakage analysis via orthonormal decomposition.
              In Innovative Security Solutions for Information Technology and Communications: 10th International Conference, SecITC 2017, Bucharest,
              Romania, June 8–9, 2017, Revised Selected Papers 10, pages 12–27. Springer, 2017.

    Arguments:
        targeted_values: set of targeted variables (i.e. Sbox output of the xor of plaintexts and the key)
        max_nb_monomials_interactions : maximal degree of bit interactions. If targeted values are bytes, max_nb_monomials_interactions=8
        len_basis: number of monomials considered. Usually, len_basis is set to 2**max_nb_monomials_interactions (all monomials are considered).
                   But it can also be set to another value:
                    (maximal degree of bit interactions=0 => len_basis=1 /
                    maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                    maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                    maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                    maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)

    Returns:
        Projection of target values in Guilley etal. orthonormal basis.
    """

    len_targeted_values = targeted_values.shape[0]
    max_possible_values = 2**max_nb_monomials_interactions

    u = np.asarray(
        [(i, bin(i)[2:].count("1")) for i in range(max_possible_values)],
        dtype=[("val", int), ("hw", int)],
    )
    u_increasing_hw = np.sort(u, order="hw")
    u_increasing_hw = u_increasing_hw["val"]

    # Computation of the Fourier basis using Hadamard matrix [GHMR17, Section 3.2]
    walsh_hadamard_matrix = (1 / 2 ** (max_nb_monomials_interactions / 2)) * hadamard(
        max_possible_values
    )

    # Projection of target values in Fourier basis [GHMR17, Theorem 5]
    basis_projection = np.zeros((len_targeted_values, max_possible_values))
    for i in range(len_targeted_values):
        basis_projection[i, :] = walsh_hadamard_matrix[targeted_values[i], :]

    # Projection sorted by increasing Hamming weight [GHMR17, Section 3.2]
    basis_projection_sorted = basis_projection[:, u_increasing_hw]

    return basis_projection_sorted[:, :len_basis]



# =====================================================================
#
# ----------------------- Generation of traces ------------------------
#
# =====================================================================

def generate_traces(
    nb_traces,
    nb_samples,
    nb_poi=1,
    mu=0,
    isotropic_noise=False,
    sigmas=np.array([1e-1]),
    alpha=np.array([1, 0.5, 1, 0, 2, 0.5, 0.75, 0.25]),
    beta=np.array(
        [1, 2, 0.5, 0.25, 1, 0.15, 1, 0.4, 0.3, 0.6, 0.2, 0.1, 0.25, 0.5, 0.75, 1]
    ),
    scenario="scenario_4",
    targeted_variables_type='aes_sbox',
    targeted_variables_size = 8,
    targeted_byte=0,
    key=[0x4a, 0x58, 0x32, 0xae, 0x1f, 0x02, 0x96, 0xe1, 0xcc, 0x3d, 0xb4, 0x13, 0xaa, 0x8c, 0xf6, 0xa7],
    seed=42
):
    """
    Generation of nb_traces traces according to the selected scenario given a fixed key.

    Arguments:
        nb_traces: number of traces to generate
        nb_samples: number of samples per generated trace
        nb_poi: number of points of interest per generated trace
        mu: mean of the multivariate Gaussian distribution which is followed by the white gaussian noise
        isotropic_noise: boolean that specifies if the Guassian noise is isotropic
        sigmas: array of variances of the multivariate Gaussian distribution.
            If the noise is isotropic, len(sigmas)==1.
            Else, noise variances will be randomly chosen among the values in sigmas (hence
            sigmas can have a length smaller than the samples argument).
        alpha: coefficients associated with bits leakage (Independent Bit Leakage Model)
        beta: coefficients associated with bits leakage (Multivariate Leakage)
        scenario: name of the scenario used to generate traces
                  - "scenario_1": Hamming Weight leakage model (HW)
                  - "scenario_2": Independent Bit leakage model (IBL)
                  - "scenario_3": Multivariate leakage model, i.e. we consider bits interactions
                  - "scenario_4": Multi leakage model (HW, IBL and multivariate leakage model)
        targeted_variables_type: type of the targeted variables
                  - "aes_sbox": Output of AES Sbox
                  - "prince_sbox": Output of Prince Sbox
                  - "clyde_sbox": Output of Clyde128 Sbox
                  - "xor": XOR between plaintexts and key
        targeted_variables_size: size (number of bits) of targeted variables. 
                  - if the targeted variables are bytes, targeted_variables_size=8 
                  - if the targeted variables consist in 4 bits, targeted_variables_size=4
        targeted_byte: targeted byte/bit
        key : fixed key used to generate traces
        seed: value of seed for reproductible results. To disable reproductible results option, seed must be set to None

    Returns:
        A tuple composed of:
        - traces: generated traces;
        - plaintexts: used plaintexts;
        - targeted_variables: targeted variables;
        - sigmas_vector: values of sigmas used to generate these traces.
    """

    # Set seed
    np.random.seed(seed)

    # Initialization of the generated traces
    if isotropic_noise:
        if not isinstance(sigmas, list):
            sigmas = [sigmas]
        assert len(sigmas) == 1
        sigmas_vector = np.array([sigmas[0] for i in range(nb_samples)])
    else:
        sigmas_vector = np.random.choice(sigmas, nb_samples)

    # Generation of the multivariate Gaussian noise
    traces = np.random.multivariate_normal(
        np.array([mu for i in range(nb_samples)]),
        (sigmas_vector**2) * np.identity(nb_samples),
        size=(nb_traces),
    )

    # Initialization of the data (i.e. plaintexts, targeted_variables)
    plaintexts = np.random.randint(0, 2**targeted_variables_size, (nb_traces,), np.uint8)

    # Computation of the targeted variable (ie. Output Sbox)
    match targeted_variables_type:
        case 'aes_sbox':
            targeted_variables = aes_sbox_vectorized(plaintexts, key[targeted_byte])
        
        case 'prince_sbox':
            targeted_variables = prince_sbox_vectorized(plaintexts, key[targeted_byte])            

        case 'clyde_sbox':
            targeted_variables = clyde_sbox_vectorized(plaintexts, key[targeted_byte])

        case 'xor':
            targeted_variables = plaintexts ^ key[targeted_byte]
        
        case _:
            raise Exception("[Targeted variables computation] - Undefined cryptographic functions")

    # Construction of the simulated traces
    monomials_deg_1 = np.unpackbits(
        targeted_variables.astype("uint8"), bitorder="little"
    ).reshape(nb_traces, 8)[:,:targeted_variables_size]

    # Computing leakage models
    ## HW
    HW_Y = np.sum(monomials_deg_1, axis=1)

    ## IBL
    IBL_Y = np.sum(monomials_deg_1 * alpha, axis=1)

    ## Multivariate
    match targeted_variables_type:
        case 'aes_sbox' | 'xor':
            leakage = np.array([monomials_deg_1[:,0], monomials_deg_1[:,5], monomials_deg_1[:,6], \
           
                monomials_deg_1[:,1] ^ monomials_deg_1[:,3], monomials_deg_1[:,2] ^ monomials_deg_1[:,4], \
                monomials_deg_1[:,4] ^ monomials_deg_1[:,7], \
                
                monomials_deg_1[:,0] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,6], monomials_deg_1[:,1] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,7],\
                monomials_deg_1[:,1] ^ monomials_deg_1[:,6] ^ monomials_deg_1[:,7],\
                
                monomials_deg_1[:,2] ^ monomials_deg_1[:,3] ^ monomials_deg_1[:,4] ^ monomials_deg_1[:,6],\
                monomials_deg_1[:,3] ^ monomials_deg_1[:,4] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,7],\
                
                monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,6],\
                monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,7],\
                
                monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,6] ^ monomials_deg_1[:,7],\
                
                monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,3] ^ monomials_deg_1[:,5] ^ monomials_deg_1[:,6] \
                ^ monomials_deg_1[:,7],\
                
                monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,3] ^ monomials_deg_1[:,4] ^ monomials_deg_1[:,5] \
                ^ monomials_deg_1[:,6] ^ monomials_deg_1[:,7]]).T

        case 'prince_sbox' | 'clyde_sbox':
            leakage = np.array([monomials_deg_1[:,0], monomials_deg_1[:,1], monomials_deg_1[:,3], \
           
                monomials_deg_1[:,0] ^ monomials_deg_1[:,1], monomials_deg_1[:,0] ^ monomials_deg_1[:,3], \
                monomials_deg_1[:,1] ^ monomials_deg_1[:,2], monomials_deg_1[:,1] ^ monomials_deg_1[:,3], \
                monomials_deg_1[:,2] ^ monomials_deg_1[:,3], \
                
                monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2], monomials_deg_1[:,0] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,3],\
                monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,3],\
                
                monomials_deg_1[:,0] ^ monomials_deg_1[:,1] ^ monomials_deg_1[:,2] ^ monomials_deg_1[:,3] ]).T

        case _:
            raise Exception('[Multivariate leakage model] - Undefined cryptographic functions')
    
    multivariate_leakage_Y = np.sum(leakage*beta,axis=1)

    # Computing scenarios
    match scenario:
        case "scenario_1":
            # Leakage model (HW only)
            leakages = [HW_Y]

        case "scenario_2":
            # Leakage model (IBL only)
            leakages = [IBL_Y]

        case "scenario_3":
            # Leakage model (multivariate leakage only)
            leakages = [multivariate_leakage_Y]

        case "scenario_4":
            # Leakage model (HW, IBL and multivariate leakage depending on PoIs)
            leakages = [HW_Y, IBL_Y, multivariate_leakage_Y]

        case _:
            raise Exception("Undefined scenario")

    # Insertion of the leakage model in the simulated traces
    # if more than a single leakage model, then PoI l will use
    # the leakage model at position l%len(leakages)
    for l in range(nb_poi):
        if nb_poi == nb_samples:
            traces[:, l] += leakages[l % len(leakages)]
        else:
            traces[:, (int(nb_samples / (nb_poi + 1)) * (l + 1))] += leakages[
                l % len(leakages)
            ]

    return traces, plaintexts, targeted_variables, sigmas_vector
