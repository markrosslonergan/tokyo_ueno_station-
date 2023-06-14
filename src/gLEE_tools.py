#!/usr/bin/env python3
import numpy as np
import uproot as up
import awkward as ak
import pandas as pd



def loadgLEE(filename):
    file = up.open(filename)
    vertex = file["singlephoton/vertex_tree"]
    simple = file["singlephoton/simple_tree"]
    #grab pandas dataframes of useful info only, alias a few things for simplicity
    ary1 = vertex.arrays(["reco_asso_showers","reco_asso_tracks","m_flash_optfltr_pe_beam","m_flash_optfltr_pe_veto","reco_vertex_x", "reco_vertex_y", "reco_vertex_z","true_vertex_x","true_vertex_y","true_vertex_z", "reco_primary_shower_energy","reco_secondary_shower_energy","reco_track_energy","true_energy_e_minus","true_energy_sum","true_mom_e_plus","true_mom_e_minus","true_delta_theta","true_pz_p_e_plus","true_pz_p_e_minus","true_phi_e_minus","true_pos_decay_z"], 
                  aliases={"true_vertex_x": "mctruth_daughters_startx[:,5]",
                           "true_vertex_y":"mctruth_daughters_starty[:,5]",
                           "true_vertex_z":"mctruth_daughters_startz[:,5]",
                           "reco_primary_shower_energy" : "(reco_shower_energy_plane2[:,0]+reco_shower_energy_max[:,0]*(reco_shower_energy_plane2[:,0]==0))*1.21989 +8.50486 ",
                           "reco_secondary_shower_energy" : "(reco_shower_energy_plane2[:,1]+reco_shower_energy_max[:,1]*(reco_shower_energy_plane2[:,1]==0))*1.21989 +8.50486 ",
                           "reco_track_energy" : "(reco_track_calo_energy_plane2[:,0]+reco_track_calo_energy_max[:,0]*(reco_track_calo_energy_plane2[:,0]==0))*1.21989 +8.50486 ",
                           #The observables needed for reweighting
                           "true_energy_e_minus" : "mctruth_daughters_E[:,5]",
                           "true_energy_e_plus" : "mctruth_daughters_E[:,6]",
                           "true_energy_sum" : "(mctruth_daughters_E[:,5]+mctruth_daughters_E[:,6])",
                           "true_mom_e_minus" : "sqrt(mctruth_daughters_px[:,5]*mctruth_daughters_px[:,5]+mctruth_daughters_py[:,5]*mctruth_daughters_py[:,5]+mctruth_daughters_pz[:,5]*mctruth_daughters_pz[:,5])",
                           "true_mom_e_plus" : "sqrt(mctruth_daughters_px[:,6]*mctruth_daughters_px[:,6]+mctruth_daughters_py[:,6]*mctruth_daughters_py[:,6]+mctruth_daughters_pz[:,6]*mctruth_daughters_pz[:,6])",
                           "true_delta_theta" : "acos((mctruth_daughters_px[:,5]*mctruth_daughters_px[:,6]+mctruth_daughters_py[:,5]*mctruth_daughters_py[:,6]+mctruth_daughters_pz[:,5]*mctruth_daughters_pz[:,6])/(true_mom_e_minus*true_mom_e_plus))/3.14159*180.",
                           "true_pz_p_e_minus" : "mctruth_daughters_pz[:,5]/true_mom_e_minus",
                           "true_pz_p_e_plus" : "mctruth_daughters_pz[:,6]/true_mom_e_plus",
                           "true_phi_e_minus" : "atan2(mctruth_daughters_py[:,5],mctruth_daughters_px[:,5])/3.14159*180.0", 
                           "true_phi_e_plus" : "atan2(mctruth_daughters_py[:,6],mctruth_daughters_px[:,6])/3.14159*180.0",
                           "true_pos_decay_z" : "mctruth_daughters_startz[:,5]-518.4" }, library="pd")

    #grab the BDT score pass/fail
    #First one seems to break with periods in the name, hmm
    #ary2 = simple.arrays(["pass_selection"],aliases={"pass_selection": "(simple_DarkNu_MultiTop_v2.5COSMIC_mva > 0.994)*(simple_DarkNu_MultiTop_v2.5Nue_mva > 0.983 )*(simple_DarkNu_MultiTop_v2_5BNB_mva > 0.987 )*(simple_DarkNu_MultiTop_v2_5NCPi0_mva > 0.8377) "}, library="pd")
    #Try this instead
    
    ary2 = simple.arrays(['simple_weight','simple_pot_weight','simple_MultiTop_v5_0COSMIC_mva','simple_MultiTop_v5_0Nue_mva','simple_MultiTop_v5_0BNB_mva','simple_MultiTop_v5_0NCPi0_mva'], library="pd")
    ary2.columns = ary2.columns.str.replace(".","_", regex=False)
    ary2['pass_selection'] = ary2.apply(lambda row: (row['simple_MultiTop_v5_0COSMIC_mva']>0.994)*(row['simple_MultiTop_v5_0Nue_mva']>0.983)*(row['simple_MultiTop_v5_0BNB_mva']>0.987)*(row['simple_MultiTop_v5_0NCPi0_mva']>0.8377), axis=1)
     
    df = pd.concat([ary1, ary2], axis=1)
    df.fillna(0,inplace=True)

    #energy is just the sum of all reconstructed objects
    df['reco_energy'] = df.apply(lambda row: row['reco_primary_shower_energy'] + row['reco_secondary_shower_energy']+row['reco_track_energy'], axis=1)
    return df

#Notes
#           - Energy e- (true_energy_e_minus)
#           - Total energy of e+e- pair (true_energy_sum)
#           - Delta theta (true_delta_theta): separation angle of e+e- pair
#           - pz/p for e+ (true_pz_p_e_plus)
#           - pz/p for e- (true_pz_p_e_minus)
#           - azimutal angle phi for e- (true_phi_e_minus): from positive x-axis direction counterclockwise

def loadgLEE_Bare(filename):
    file = up.open(filename)
    vertex = file["singlephotonana/vertex_tree"]
    #grab pandas dataframes of useful info only, alias a few things for simplicity
    ary1 = vertex.arrays(["true_vertex_x","true_vertex_y","true_vertex_z", "true_energy_e_minus","true_energy_sum","true_mom_e_plus","true_mom_e_minus","true_delta_theta","true_pz_p_e_plus","true_pz_p_e_minus","true_phi_e_minus","textgen_weight","true_pos_decay_z"], 
                  aliases={"true_vertex_x": "mctruth_daughters_startx[:,5]",
                           "true_vertex_y":"mctruth_daughters_starty[:,5]",
                           "true_vertex_z":"mctruth_daughters_startz[:,5]",
                           #The observables needed for reweighting
                           "true_energy_e_minus" : "mctruth_daughters_E[:,5]",
                           "true_energy_e_plus" : "mctruth_daughters_E[:,6]",
                           "true_energy_sum" : "(mctruth_daughters_E[:,5]+mctruth_daughters_E[:,6])",
                           "true_mom_e_minus" : "sqrt(mctruth_daughters_px[:,5]*mctruth_daughters_px[:,5]+mctruth_daughters_py[:,5]*mctruth_daughters_py[:,5]+mctruth_daughters_pz[:,5]*mctruth_daughters_pz[:,5])",
                           "true_mom_e_plus" : "sqrt(mctruth_daughters_px[:,6]*mctruth_daughters_px[:,6]+mctruth_daughters_py[:,6]*mctruth_daughters_py[:,6]+mctruth_daughters_pz[:,6]*mctruth_daughters_pz[:,6])",
                           "true_delta_theta" : "acos((mctruth_daughters_px[:,5]*mctruth_daughters_px[:,6]+mctruth_daughters_py[:,5]*mctruth_daughters_py[:,6]+mctruth_daughters_pz[:,5]*mctruth_daughters_pz[:,6])/(true_mom_e_minus*true_mom_e_plus))/3.14159*180.",
                           "true_pz_p_e_minus" : "mctruth_daughters_pz[:,5]/true_mom_e_minus",
                           "true_pz_p_e_plus" : "mctruth_daughters_pz[:,6]/true_mom_e_plus",
                           "true_phi_e_minus" : "atan2(mctruth_daughters_py[:,5],mctruth_daughters_px[:,5])/3.14159*180.0", 
                           "true_phi_e_plus" : "atan2(mctruth_daughters_py[:,6],mctruth_daughters_px[:,6])/3.14159*180.0" ,
                           "textgen_weight" : "textgen_info[:,0]" ,
                           "true_pos_decay_z" : "mctruth_daughters_startz[:,5]-518.4"}, library="pd")
    df = ary1
    df.fillna(0,inplace=True)

    return df






def getWeight(bp_mapBench,bp_mapTarget,input_values):
    #input values is a list of parameters that we reweight/bin in 
    
    binlist = np.zeros(bp_mapBench.attrs["observables"].size,dtype=int)
    
    for obs,val,index in zip(bp_mapBench.attrs["observables"],input_values, enumerate(binlist)) :
        #searchsorted returns -1 if below, and will return size+1 above, 
        #can assume binning is always sorted?
        tbin = np.searchsorted(bp_mapBench.attrs[obs], val )-1
        #unified -1 if value is outside binning
        tbin = tbin if tbin < bp_mapBench.attrs[obs].size-1 else -1   
        binlist[index[0]]=tbin
    #is it a valid value?
    valid = (binlist >=0).all()
    
    if valid:
        #print(bp_mapBench["map"][binlist])
        valBench = bp_mapBench["map"][binlist[0],binlist[1],binlist[2],binlist[3],binlist[4],binlist[5]]
        valTarget =bp_mapTarget["map"][binlist[0],binlist[1],binlist[2],binlist[3],binlist[4],binlist[5]]
        return valTarget/valBench if valBench!=0 else -1e-10

    else :
        return -999
    
def getMasterWeight(map_Ap, map_Bp, map_C, map_D, bp_mapTarget,input_values,map_corr) :
   
    binlist = np.zeros(bp_mapTarget.attrs["observables"].size,dtype=int)
    
    for obs,val,index in zip(bp_mapTarget.attrs["observables"],input_values, enumerate(binlist)) :
        #searchsorted returns -1 if below, and will return size+1 above, 
        #can assume binning is always sorted?
        tbin = np.searchsorted(bp_mapTarget.attrs[obs], val )-1
        #unified -1 if value is outside binning
        tbin = tbin if tbin < bp_mapTarget.attrs[obs].size-1 else -1   
        binlist[index[0]]=tbin
    
    #print(binlist)
    #is it a valid value?
    valid = (binlist >=0).all()
    
    if valid:
        #print(bp_mapBench["map"][binlist])
        valA = map_Ap["map"][binlist[0],binlist[1],binlist[2],binlist[3],binlist[4],binlist[5],binlist[6]]/map_corr[0]
        valB = map_Bp["map"][binlist[0],binlist[1],binlist[2],binlist[3],binlist[4],binlist[5],binlist[6]]/map_corr[1]
        valC = map_C["map"][binlist[0],binlist[1],binlist[2],binlist[3],binlist[4],binlist[5],binlist[6]]/map_corr[2]
        valD = map_D["map"][binlist[0],binlist[1],binlist[2],binlist[3],binlist[4],binlist[5],binlist[6]]/map_corr[3]
        valBench = valA+valB+valD+valC
        
        valTarget =bp_mapTarget["map"][binlist[0],binlist[1],binlist[2],binlist[3],binlist[4],binlist[5],binlist[6]]
        return valTarget/valBench if valBench!=0 else  0 # -1e-10

    else :
        return 0 #-999