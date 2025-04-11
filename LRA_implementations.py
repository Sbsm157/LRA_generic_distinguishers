import numpy as np
from tqdm import tqdm
from .LRA_utils import orthonormal_basis_projection

# =====================================================================
#
# -------------------- Distinguishers computation ---------------------
#
# =====================================================================

def compute_distinguisher(distinguisher, scores):
    """
    Computation of distinguishers.

    Arguments:
        distinguisher: chosen distinguisher to perform LRA based attacks
            - "R2_score": R2 distinguisher
            - "R2_normalized_score": Normalized R2 distinguisher [LPR13, Section 3.2]
            - "maximum_distinguisher": Maximum distinguisher (see Section 4.1.3, Proposition 4 in our paper)
            - "absolute_value_sum_distinguisher": Absolute value of the sum distinguisher distinguisher (see Section 4.2, Proposition 6 in our paper)
        scores: R2 scores in case of R2-based distinguishers are chosen. 
                If maximum or absolute value of the sum distinguisher is chosen, 
                it corresponds to coefficients estimation during LRA process

    [LPR13]  Lomné, V., Prouff, E., Roche, T.: Behind the scene of side channel attacks. 
             In: Sako, K., Sarkar, P. (eds.) ASIACRYPT 2013, Part I. LNCS, vol. 8269, pp. 506–525. Springer, Heidelberg (2013)

    Returns:
        Keys ranking according to the selected distinguisher
    """   

    # Computation of distinguishers   
    match distinguisher:

        # Computation of R2 scores
        case "R2_score":
            keys_ranking = np.argsort(np.max(scores, axis=1))
        
        # Computation of normalized R2 scores (see [LPR13, Section 3.2])
        case "R2_normalized_score":
            mu_R2_scores = np.mean(scores, axis=0)
            sigma_R2_scores = np.mean((scores - mu_R2_scores)**2, axis=0)
            max_u = np.max(scores, axis=0)
            argmax_u = np.argsort(scores, axis=0)
            normalized_max = (max_u - mu_R2_scores)/sigma_R2_scores
            keys_ranking = argmax_u[:, np.argmax(normalized_max)]
        
        # Computation of maximum distinguisher (see Section 4.1.3, Proposition 4 in our paper)
        case "maximum_distinguisher":
            keys_ranking = np.argsort(np.max(np.abs(scores[:,1:,:]), axis=(1,2)))
        
        # Computation of absolute value of the sum distinguisher distinguisher (see Section 4.2, Proposition 6 in our paper)
        case "absolute_value_sum_distinguisher":
            keys_ranking = np.argsort(np.max(np.abs(np.sum(scores[:,1:,:], axis=1)),axis=1))
        
        case _:
            raise Exception("Undefined LRA distinguisher")
    
    return keys_ranking[::-1]



# =====================================================================
#
# ------------------------ LRA implementations ------------------------
#
# =====================================================================

def LRA(traces, plaintexts, nb_hypothesis, len_basis, monomials_max_degree, targeted_f_type, sbox):
    """
    Implementation of the Linear Regression Analysis algorithm introduced by Lomné etal. in [LPR13, Algorithm 3].

    [LPR13]  Lomné, V., Prouff, E., Roche, T.: Behind the scene of side channel attacks. 
             In: Sako, K., Sarkar, P. (eds.) ASIACRYPT 2013, Part I. LNCS, vol. 8269, pp. 506–525. Springer, Heidelberg (2013)

    Arguments:
        traces: attack traces
        plaintexts: attack plaintexts
        nb_hypothesis: key hypotheses
        len_basis: number of monomials considered. Usually, len_basis is set to 2**max_nb_monomials_interactions (all monomials are considered).
                   But it can also be set to another value:
                    (maximal degree of bit interactions=0 => len_basis=1 /
                    maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                    maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                    maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                    maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)
        monomials_max_degree : maximal degree of bit interactions. If targeted values are bytes, max_nb_monomials_interactions=8
        targeted_f_type: type of the targeted cryptographic function f
            - 'sbox': Sbox function
            - 'xor': xor operation
        sbox: Sbox function if targeted_f_type='sbox'. If targeted_f_type='xor', sbox is set to None. 

    Returns:
        - Estimated coefficients for all key hypotheses
        - R2 scores for all key hypotheses
    """

    # Computation of leakage Total Sum of Squares
    mu_l = np.sum(traces, axis=0)
    sigma_l = np.sum(traces**2, axis=0)
    SST = sigma_l - mu_l**2/traces.shape[0]
    
    # Initialization of matrices
    P = np.zeros((nb_hypothesis, len_basis, traces.shape[0]))
    M = np.zeros((nb_hypothesis, traces.shape[0], len_basis))
    R2_scores = np.zeros((nb_hypothesis, traces.shape[1]))
    
    coefficients = []
    
    # Computation of the predictions matrices
    for k in range(nb_hypothesis):
        if targeted_f_type == 'sbox': 
            y = sbox(plaintexts, k) 
        else:
            y = plaintexts ^ k 

        # Projection of targeted variables into WH basis
        M[k] = orthonormal_basis_projection(y, monomials_max_degree, len_basis)

        # Computation of prediction matrices for a key hypothesis k 
        try:
            P[k] = np.linalg.inv(M[k].T @ M[k]) @ M[k].T
        except: # cases where (M[k].T @ M[k]) is not inversible
            P[k] = np.linalg.pinv(M[k].T @ M[k]) @ M[k].T 
        
        # Coefficients estimation
        beta = P[k] @ traces
        coefficients.append(beta)

        # Computation of estimators
        eps = M[k] @ beta

        # Computation of estimation errors
        SSR = np.sum((eps-traces)**2, axis=0)

        # Computation of R2 scores
        R2_scores[k] = 1 - SSR/SST

    return np.array(coefficients), R2_scores

def LRA_coalesced(traces, plaintexts, nb_possible_plaintexts_values, nb_hypothesis, len_basis, monomials_max_degree, targeted_f_type, sbox):
    """
    Implementation of the coalesced Linear Regression Analysis algorithm using averaged leakages trick introduced by Lomné etal. in [LPR13, Section 3.3].

    [LPR13]  Lomné, V., Prouff, E., Roche, T.: Behind the scene of side channel attacks. 
             In: Sako, K., Sarkar, P. (eds.) ASIACRYPT 2013, Part I. LNCS, vol. 8269, pp. 506–525. Springer, Heidelberg (2013)

    Arguments:
        traces: attack traces
        plaintexts: attack plaintexts
        nb_possible_plaintexts_values: number of possible plaintexts values. If plaintexts are bytes, nb_possible_plaintexts_values=256
        nb_hypothesis: key hypotheses
        len_basis: number of monomials considered. Usually, len_basis is set to 2**max_nb_monomials_interactions (all monomials are considered).
                   But it can also be set to another value:
                    (maximal degree of bit interactions=0 => len_basis=1 /
                    maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                    maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                    maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                    maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)
        monomials_max_degree : maximal degree of bit interactions. If targeted values are bytes, max_nb_monomials_interactions=8
        targeted_f_type: type of the targeted cryptographic function f
            - 'sbox': Sbox function
            - 'xor': xor operation
        sbox: Sbox function if targeted_f_type='sbox'. If targeted_f_type='xor', sbox is set to None. 

    Returns:
        - Estimated coefficients for all key hypotheses
        - R2 scores for all key hypotheses
    """  

    # Computation of leakage Total Sum of Squares
    mu_l = np.sum(traces, axis=0)
    sigma_l = np.sum(traces**2, axis=0)
    SST = sigma_l - mu_l**2/traces.shape[0]

    # Averaged leakages (coalesced LRA trick) see [LPR13, Section 3.3]
    traces_avg = np.array([np.mean(traces[np.where(plaintexts==p)[0],:], axis=0) for p in range(nb_possible_plaintexts_values)])
    
    # Initialization of matrices
    P = np.zeros((nb_hypothesis, len_basis, traces_avg.shape[0]))
    M = np.zeros((nb_hypothesis, traces_avg.shape[0], len_basis))
    R2_scores = np.zeros((nb_hypothesis, traces_avg.shape[1]))
    
    coefficients = []

    # Computation of the predictions matrices
    for k in range(nb_hypothesis): 
        if targeted_f_type == 'sbox': 
            y = sbox(np.array([i for i in range(nb_possible_plaintexts_values)]), k) 
        else:
            y = np.array([i for i in range(nb_possible_plaintexts_values)]) ^ k

        # Projection of targeted variables into WH basis
        M[k] = orthonormal_basis_projection(y, monomials_max_degree, len_basis)

        # Computation of prediction matrices for a key hypothesis k
        try:
            P[k] = np.linalg.inv(M[k].T @ M[k]) @ M[k].T
        except: # cases where (M[k].T @ M[k]) is not inversible
            P[k] = np.linalg.pinv(M[k].T @ M[k]) @ M[k].T

        # Coefficients estimation
        beta = P[k] @ traces_avg
        coefficients.append(beta)

        # Computation of estimators
        eps = M[k] @ beta

        # Computation of estimation errors
        SSR = np.sum((eps-traces_avg)**2, axis=0)

        # Computation of R2 scores
        R2_scores[k] = 1 - SSR/SST
        
    return np.array(coefficients), R2_scores



# =====================================================================
#
# ----------------------- Averaged LRA attacks ------------------------
#
# =====================================================================

def LRA_averaged_attacks(total_nb_attack, LRA_attack, true_key, nb_traces, attack_traces, attack_plaintexts, len_basis, deg_monomials_interactions, \
                             nb_possible_plaintexts_values, nb_hypothesis, targeted_f_type, sbox, distinguisher):
    """
     Runs *total_nb_attack* LRA based attacks.

    Arguments:
        total_nb_attack: total number of attacks to carry out
        LRA_attack: type of LRA that is conducted
                    - 'LRA': LRA based attacks
                    - 'LRA_coalesced': coalesced LRA based attacks
        true_key: real key value
        nb_traces: array that contains numbers of traces on which we conduct attacks
        attack_traces: set of attack traces
        attack_plaintexts: set of attack plaintexts
        len_basis: number of monomials considered. Usually, len_basis is set to 2**max_nb_monomials_interactions (all monomials are considered).
                        But it can also be set to another value:
                            (maximal degree of bit interactions=0 => len_basis=1 /
                            maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                            maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                            maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                            maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)
                monomials_max_degree : maximal degree of bit interactions. If targeted values are bytes, max_nb_monomials_interactions=8
        deg_monomials_interactions: maximal degree of bit interactions. If targeted values are bytes, max_nb_monomials_interactions=8
        nb_possible_plaintexts_values: number of possible plaintexts values. If plaintexts are bytes, nb_possible_plaintexts_values=256
        nb_hypothesis: key hypotheses
        targeted_f_type: type of the targeted cryptographic function f
                    - 'sbox': Sbox function
                    - 'xor': xor operation
        sbox: Sbox function if targeted_f_type='sbox'. If targeted_f_type='xor', sbox is set to None. 
        distinguisher: chosen distinguisher among our proposed distinguishers to perform LRA based attacks
                    - "maximum_distinguisher": Maximum distinguisher (see Section 4.1.3, Proposition 4 in our paper)
                    - "absolute_value_sum_distinguisher": Absolute value of the sum distinguisher distinguisher (see Section 4.2, Proposition 6 in our paper)

    Returns:
        Mean evolutions of key rank as a function of nb_traces, considering R2 distinguisher, normalized R2 distinguisher and our distinguisher 
        (either maximum or absolute valu of the sum distinguisher according to the targeted cryptographic function)
    """ 

    # Initialization 
    indexs_traces = np.arange(attack_traces.shape[0])
    offset_departure_subsets_traces = attack_traces.shape[0] // total_nb_attack
    index_traces_attacks = []
    key_rank = np.zeros((len(nb_traces), total_nb_attack))
    key_rank_R2 = np.zeros((len(nb_traces), total_nb_attack))
    key_rank_norm_R2 = np.zeros((len(nb_traces), total_nb_attack))

    # Setting attack trace indexes for each attack
    for current_attack in range(total_nb_attack):
            index_traces_attacks.append(np.roll(indexs_traces, offset_departure_subsets_traces * current_attack))

    # Running of the LRA based attacks on all numbers of traces
    for i in tqdm(range(len(nb_traces))):
        
        # Running of the LRA based attack *total_nb_attack* times considering *nb_traces[i]* traces
        for j in range(total_nb_attack):

            # Extraction of the proper attack traces subset
            index_traces_current_attack = index_traces_attacks[j]
            current_subset_index = index_traces_current_attack[:nb_traces[i]]
            traces_attack_subset = attack_traces[current_subset_index, :]
            plaintexts_attack_subset = attack_plaintexts[current_subset_index]

            # Run of the LRA based attack (either LRA or coalesced LRA)
            if LRA_attack == 'LRA':
                coefficients, R2_scores = LRA(traces_attack_subset, plaintexts_attack_subset, nb_hypothesis, len_basis,  deg_monomials_interactions, targeted_f_type, sbox)
            else: 
                coefficients, R2_scores = LRA_coalesced(traces_attack_subset, plaintexts_attack_subset, nb_possible_plaintexts_values, \
                                                            nb_hypothesis, len_basis, deg_monomials_interactions, targeted_f_type, sbox)

            # Computation of the rank for the current attack considering R2 distinguisher, normalized R2 distinguisher and our distinguisher 
            # (either maximum or absolute value of the sum distinguisher according to the targeted cryptographic function)
            keys_ranking = compute_distinguisher(distinguisher, coefficients)
            key_rank[i,j] = np.where(true_key == keys_ranking)[0][0]

            keys_ranking = compute_distinguisher("R2_score", R2_scores)
            key_rank_R2[i,j] = np.where(true_key == keys_ranking)[0][0]

            keys_ranking = compute_distinguisher("R2_normalized_score", R2_scores)
            try:
                key_rank_norm_R2[i,j] = np.where(true_key == keys_ranking)[0][0]
            except: # cases where the true key does not appear as key candidate among the *attack_traces.shape[1]* samples
                key_rank_norm_R2[i,j] = nb_hypothesis // 2
    
    return np.mean(key_rank, axis=1), np.mean(key_rank_R2, axis=1),  np.mean(key_rank_norm_R2, axis=1)
    