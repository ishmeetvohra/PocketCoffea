import awkward as ak
import numpy as np

# cc_score for Trevor
# CC_score for Ishmeet 
def get_dijet_paired(
    events,
    jet_coll="Jet",
    jetgood_mask=None,
    pair_table="PAIReDJets",
    score_field="CC_score", 
    taggerVars = True,
    jet_tagger = "",
    return_remnant_jet = True
):
    """
    Build Higgs candidate from best PAIReD jet-pair (max score_field (CC score or BB score)),
    and remnant = sum of JetGood jets excluding the two chosen jets.

    Parameters
    ----------
    events : NanoEventsArray
    jet_coll : str
        Name of the jet collection: "Jet"
    jetgood_mask : awkward Array (bool)
        Mask on events[jet_coll] defining JetGood (same shape as jets).
        This is self.jetGoodMask returned by jet_selection
    pair_table : str
        FlatTable name containing idx_jet1,idx_jet2 and scores
    score_field : str
        Which score to maximize when choosing best pair
    taggerVars : bool
        If True and reco-jet tagger fields exist, fill j1CvsL/j2CvsL/j1CvsB/j2CvsB
    jet_tagger : str
        AK4 jet tagger used to access the corresponding tagger branches (may be useful for comparison studies of the selected jet pairs)
    return_remnant : bool
        If True also return sum of remaining JetGood jets for kinematic fit

    Returns
    -------
    dijet : (nEvents,) PtEtaPhiMCandidate-like record with extra fields (j1pt, j2pt, deltaR, ...)
    remnant : (nEvents,) PtEtaPhiMCandidate-like record sum of remaining JetGood jets excluding the 2 higgs candidate jets
    best_pair : (nEvents,) record: idx_jet1, idx_jet2, best_score, other scores for the best pair, best pair objects
    """

    if jetgood_mask is None:
        raise ValueError("jetgood_mask is required (mask on events[jet_coll]).")

    jets = events[jet_coll] #per event jagged jet collection

    if pair_table not in events.fields:
        raise KeyError(f"Pair table '{pair_table}' not found in events.fields")

    pairedtable = events[pair_table] #per event jagged table of candidate jet pairs containing indices of jet1 and jet2 of the pair and the 6 tagger scores

    #indices into the original jet collection 
    idx1 = ak.values_astype(pairedtable["idx_jet1"], np.int64) #jagged array (per event per pair)
    idx2 = ak.values_astype(pairedtable["idx_jet2"], np.int64)
    
    
    '''
    print("jets type:", ak.type(jets))
    print("mask type:", ak.type(jetgood_mask))
    print("idx1 type:", ak.type(idx1))
    #print("pairs type:", ak.type(pairedtable[score_field]))
    '''

    #keep only pairs where both jets are JetGood
    good_pairs = jetgood_mask[idx1] & jetgood_mask[idx2]
    
    idx1g = idx1[good_pairs]
    idx2g = idx2[good_pairs]
    
    score = pairedtable[score_field][good_pairs]

    npairs = ak.num(idx1g) #counts the number of good pairs per event
    has = npairs > 0 #event level bool mask for identifying if event has at least one good pair

    #choose best pair per event
    best_k = ak.argmax(score, axis=1, keepdims=True) #returns per event the index of the pair with highest CC score (this is an index into the filterd good pairs, don't use in original pairs)
    best_i1 = ak.firsts(idx1g[best_k]) #returns per event the index of the pair with highest CC score (this is an index into the filterd good pairs, don't use in original pairs)
    best_i2 = ak.firsts(idx2g[best_k]) #ak.first to resolve the list structure
    best_s  = ak.firsts(score[best_k])

    #if no pairs set to None and score to -inf 
    best_i1 = ak.fill_none(ak.mask(best_i1, has), -1)
    best_i2 = ak.fill_none(ak.mask(best_i2, has), -1)
    best_s = ak.where(has, best_s, -np.inf)
    
    all_score_fields = ["BB_score",
                        "CC_score",
                        "ll_score",
                        "bb_score",
                        "bl_score",
                        "cl_score"]
    
    best_scores = {}
    for sf in all_score_fields:
        if sf == score_field:
            continue
        if sf in pairedtable.fields:
            best_scores[sf] = ak.fill_none(
                ak.firsts(pairedtable[sf][good_pairs][best_k]),
                -999.0,
            )
    
    '''
    
    for ievt in range(min(5, len(score))):

        print(f"\n========== EVENT {ievt} ==========")

        idx1_evt = ak.to_list(idx1g[ievt])
        idx2_evt = ak.to_list(idx2g[ievt])

        cc_evt   = ak.to_list(pairedtable["CC_score"][good_pairs][ievt])
        bb_evt   = ak.to_list(pairedtable["BB_score"][good_pairs][ievt])
        bx_evt   = ak.to_list(pairedtable["BX_score"][good_pairs][ievt])
        cx_evt   = ak.to_list(pairedtable["CX_score"][good_pairs][ievt])
        ll_evt   = ak.to_list(pairedtable["LL_score"][good_pairs][ievt])
        elcc_evt = ak.to_list(pairedtable["ELCC_score"][good_pairs][ievt])
        elbb_evt = ak.to_list(pairedtable["ELBB_score"][good_pairs][ievt])
        bbnr_evt = ak.to_list(pairedtable["BB_nonres_score"][good_pairs][ievt])
        #ccvsall_evt = ak.to_list(score[ievt])


        for ipair in range(len(cc_evt)):
            print(
                f"pair {ipair:2d} "
                f"({idx1_evt[ipair]},{idx2_evt[ipair]}) | "
                f"cc={cc_evt[ipair]:.4f} "
                f"bb={bb_evt[ipair]:.4f} "
                f"bx={bx_evt[ipair]:.4f} "
                f"cx={cx_evt[ipair]:.4f} "
                f"ll={ll_evt[ipair]:.4f} "
                f"elcc={elcc_evt[ipair]:.4f} "
                f"elbb={elbb_evt[ipair]:.4f} "
                f"bbnr={bbnr_evt[ipair]:.4f}"
                #f"ccvsall={ccvsall_evt[ipair]:.4f} "
            )
            
       

        if len(cc_evt) > 0:
            sel = int(ak.firsts(best_k)[ievt])

            print("\nSELECTED PAIR")
            print(
                f"pair {sel:2d} "
                f"({idx1_evt[sel]},{idx2_evt[sel]}) | "
                f"cc={cc_evt[sel]:.4f} "
                f"bb={bb_evt[sel]:.4f} "
                f"bx={bx_evt[sel]:.4f} "
                f"cx={cx_evt[sel]:.4f} "
                f"ll={ll_evt[sel]:.4f} "
                f"elcc={elcc_evt[sel]:.4f} "
                f"elbb={elbb_evt[sel]:.4f} "
                f"bbnr={bbnr_evt[sel]:.4f}"
                #f"ccvsall={ccvsall_evt[sel]:.4f} "
            )

            print(f"\nHighest cc_score in event = {max(cc_evt):.4f}")
            #print(f"\nHighest ranking score in event = {max(ccvsall_evt):.4f}")
            
        '''

    #chosen jets
    local_idx = ak.local_index(jets.pt)
    j1 = ak.firsts(jets[local_idx == best_i1])
    j2 = ak.firsts(jets[local_idx == best_i2])
    

    '''
    CvL = None
    CvB = None
    if taggerVars:
        if "btagPNetCvL" in jets.fields and "btagPNetCvB" in jets.fields:
            CvL, CvB = "btagPNetCvL", "btagPNetCvB"
        elif "btagDeepFlavCvL" in jets.fields and "btagDeepFlavCvB" in jets.fields:
            CvL, CvB = "btagDeepFlavCvL", "btagDeepFlavCvB"
        elif "btagRobustParTAK4CvL" in jets.fields and "btagRobustParTAK4CvB" in jets.fields:
            CvL, CvB = "btagRobustParTAK4CvL", "btagRobustParTAK4CvB"
        else:
            CvL = CvB = None
    '''    
    if jet_tagger != "":
        if "PNet" in jet_tagger:
            B   = "btagPNetB"
            CvL = "btagPNetCvL"
            CvB = "btagPNetCvB"
        elif "DeepFlav" in jet_tagger:
            B   = "btagDeepFlavB"
            CvL = "btagDeepFlavCvL"
            CvB = "btagDeepFlavCvB"
        elif "RobustParT" in jet_tagger:
            B   = "btagRobustParTAK4B"
            CvL = "btagRobustParTAK4CvL"
            CvB = "btagRobustParTAK4CvB"
        else:
            raise NotImplementedError(f"This tagger is not implemented: {jet_tagger}")
        
        if B not in jets.fields or CvL not in jets.fields or CvB not in jets.fields:
            raise NotImplementedError(f"{B}, {CvL}, and/or {CvB} are not available in the input.")

        jets["btagB"] = jets[B]
        jets["btagCvL"] = jets[CvL]
        jets["btagCvB"] = jets[CvB]

            
    # enforce: j1 = leading jet in CvL, j2 = subleading jet CvL inside the chosen best pair (optional maybe interessting for comparison studies)
    """
    if CvL is not None:
        swap = j2["btagCvL"] > j1["btagCvL"]
        j1_sorted = ak.where(swap, j2, j1)
        j2_sorted = ak.where(swap, j1, j2)

        j1 = j1_sorted
        j2 = j2_sorted
    """
    
    swap = j2.pt > j1.pt
    j1_sorted = ak.where(swap, j2, j1)
    j2_sorted = ak.where(swap, j1, j2)

    j1 = j1_sorted
    j2 = j2_sorted            

    #concatenate j1 and j2 of the best pair per event 
    best_pairs = ak.concatenate([j1[:, None], j2[:, None]], axis=1)

    #dijet vector
    dijet_vec = j1 + j2

    #build dijet 
    fields = {
        "pt":   ak.where(has, dijet_vec.pt,   -1.0),
        "eta":  ak.where(has, dijet_vec.eta,  -1.0),
        "phi":  ak.where(has, dijet_vec.phi,  -1.0),
        "mass": ak.where(has, dijet_vec.mass, -1.0), 

        "deltaR":   ak.where(has, j1.delta_r(j2),            -1.0),
        "deltaPhi": ak.where(has, abs(j1.delta_phi(j2)),     -1.0),
        "deltaEta": ak.where(has, abs(j1.eta - j2.eta),      -1.0),

        "j1Phi":  ak.where(has, j1.phi,  -1.0),
        "j2Phi":  ak.where(has, j2.phi,  -1.0),
        "j1pt":   ak.where(has, j1.pt,   -1.0),
        "j2pt":   ak.where(has, j2.pt,   -1.0),
        "j1eta":  ak.where(has, j1.eta,  -1.0),
        "j2eta":  ak.where(has, j2.eta,  -1.0),
        "j1mass": ak.where(has, j1.mass, -1.0),
        "j2mass": ak.where(has, j2.mass, -1.0),
    }
    
    
    if taggerVars and jet_tagger != "":
        fields["j1CvsL"] = ak.where(has, j1["btagCvL"], -1.0)
        fields["j2CvsL"] = ak.where(has, j2["btagCvL"], -1.0)
        fields["j1CvsB"] = ak.where(has, j1["btagCvB"], -1.0)
        fields["j2CvsB"] = ak.where(has, j2["btagCvB"], -1.0)
        fields["j1BvsL"] = ak.where(has, j1["btagB"], -1.)
        fields["j2BvsL"] = ak.where(has, j2["btagB"], -1.)
    else:
        fields["j1CvsL"] = -1.0
        fields["j2CvsL"] = -1.0
        fields["j1CvsB"] = -1.0
        fields["j2CvsB"] = -1.0
        fields["j1BvsL"] = -1.0
        fields["j2BvsL"] = -1.0
        
    dijet = ak.zip(fields, with_name="PtEtaPhiMCandidate")
    
        
    
    if return_remnant_jet == True:
        #remnant = sum of JetGood excluding chosen two indices
        jetgood = jets[jetgood_mask]
        good_orig_idx = ak.local_index(jets.pt)[jetgood_mask] #original indices in full jets for the good jets


        #for no pair events keep all JetGood in remnant
        #replace nans with large negative number (invalid index)
        i1 = ak.fill_none(best_i1, -999999)
        i2 = ak.fill_none(best_i2, -999999)

        #mask False for the best jets 
        mask_remnant = (good_orig_idx != i1[:, None]) & (good_orig_idx != i2[:, None])
        remnant_jets = jetgood[mask_remnant]
        #print("remnant_jets type:", ak.type(remnant_jets))

        remnant_p4 = ak.zip(
            {"pt": remnant_jets.pt, "eta": remnant_jets.eta, "phi": remnant_jets.phi, "mass": remnant_jets.mass},
            with_name="PtEtaPhiMCandidate",
        )
        remnant_vec = ak.sum(remnant_p4, axis=-1, mask_identity=False)
        remnant = ak.zip(
            {"pt": remnant_vec.pt, "eta": remnant_vec.eta, "phi": remnant_vec.phi, "mass": remnant_vec.mass},
            with_name="PtEtaPhiMCandidate",
        )
    
    else:
        remnant = None
            

    best_pair = ak.zip(
        {
            "idx_jet1": best_i1,
            "idx_jet2": best_i2,
            "score": best_s,   
            **best_scores,
            "jets": best_pairs,
        },
        depth_limit=1,
    )
    
    return dijet, remnant, best_pair
