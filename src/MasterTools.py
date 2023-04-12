#!/usr/bin/env python3
import numpy as np
import h5py
import argparse
import shutil
import os
import os.path
from  gLEE_tools import *

import uproot as up
import matplotlib.pyplot as plt
import awkward as ak
import pandas as pd
import time
import dill
import MapBuilder as mb



class MasterTools:
    """
        Quick class to handle the master files
    """
    
    def __init__(self,indir,tag):
        self.datadir = indir
        self.tag = tag
        self.names = [self.tag+"_BPA",self.tag+"_BPD",self.tag+"_BPC",self.tag+"_BPD"]
        print("Loading Master Oriringal files @ ", self.datadir)
        self.loadMasterFinal()
        print("Loading Master Original FIles @ ", self.datadir)
        self.loadMasterOrig()
        print("Loading master DataFrames from geenration")
        self.loadMasterDFs()

    def loadMasterFinal(self):
        gLEE_dfD_fin = loadgLEE(self.datadir+"/sbnfit_DarkNu_MultiTop_v4_0_stage_1_DarkNu_KeystoneBenchD_NoHF_TextGen.root")
        gLEE_dfA_fin = loadgLEE(self.datadir+"/sbnfit_DarkNu_MultiTop_v4_0_stage_1_DarkNu_KeystoneBenchAp_NoHF_TextGen.root")
        gLEE_dfC_fin = loadgLEE(self.datadir+"/sbnfit_DarkNu_MultiTop_v4_0_stage_1_DarkNu_KeystoneBenchC_NoHF_TextGen.root")
        gLEE_dfB_fin = loadgLEE(self.datadir+"/sbnfit_DarkNu_MultiTop_v4_0_stage_1_DarkNu_KeystoneBenchBp_NoHF_TextGen.root")
        self.masterFinal = pd.concat([gLEE_dfA_fin,gLEE_dfB_fin,gLEE_dfC_fin,gLEE_dfD_fin])

    def loadMasterOrig(self):
        gLEE_dfD_vert = loadgLEE_Bare(self.datadir+"/vertex_DarkNu_Run1_Keystone_BenchD_noHF_v50.0.root")
        gLEE_dfA_vert = loadgLEE_Bare(self.datadir+"/vertex_DarkNu_Run1_Keystone_BenchAp_noHF_v50.0.root")
        gLEE_dfC_vert = loadgLEE_Bare(self.datadir+"/vertex_DarkNu_Run1_Keystone_BenchC_noHF_v50.0.root")
        gLEE_dfB_vert = loadgLEE_Bare(self.datadir+"/vertex_DarkNu_Run1_Keystone_BenchBp_noHF_v50.0.root")
        self.BP_orig = [gLEE_dfA_vert.shape[0],gLEE_dfB_vert.shape[0],gLEE_dfC_vert.shape[0],gLEE_dfD_vert.shape[0]]
        self.masterOrig = pd.concat([gLEE_dfA_vert,gLEE_dfB_vert,gLEE_dfC_vert,gLEE_dfD_vert])
         
    
    def loadMasterDFs(self):
        self.df_A = pd.read_pickle("/home/mark/work/DarkNu_Codebase_Nov2021/workin_dir/Map_Develop/microboone_study/FilteredMapGeneration/data/microboone_active_tpc_benchmark/3plus2/m5_0.15_m4_0_mzprime_0.03_dirac/pandas_df.pckl")
        self.df_B = pd.read_pickle("/home/mark/work/DarkNu_Codebase_Nov2021/workin_dir/Map_Develop/microboone_study/FilteredMapGeneration/data/microboone_active_tpc_benchmark/3plus2/m5_0.15_m4_0_mzprime_1.25_dirac/pandas_df.pckl")
        self.df_C = pd.read_pickle("/home/mark/work/DarkNu_Codebase_Nov2021/workin_dir/Map_Develop/microboone_study/FilteredMapGeneration/data/microboone_active_tpc_benchmark/3plus2/m5_0.15_m4_0.107_mzprime_0.03_dirac/pandas_df.pckl")
        self.df_D = pd.read_pickle("/home/mark/work/DarkNu_Codebase_Nov2021/workin_dir/Map_Develop/microboone_study/FilteredMapGeneration/data/microboone_active_tpc_benchmark/3plus2/m5_0.15_m4_0.107_mzprime_1.25_dirac/pandas_df.pckl")        
        BPA_corr_factor = self.df_A.shape[0]/self.BP_orig[0]
        BPB_corr_factor = self.df_B.shape[0]/self.BP_orig[1]
        BPC_corr_factor = self.df_C.shape[0]/self.BP_orig[2]
        BPD_corr_factor = self.df_D.shape[0]/self.BP_orig[3]
        print("benchmark correction factors are: ", BPA_corr_factor,BPB_corr_factor,BPC_corr_factor,BPD_corr_factor)
        self.BP_corr = [BPA_corr_factor,BPB_corr_factor,BPC_corr_factor,BPD_corr_factor]


    def buildMasterMap(self,input_binning_scheme,outputdir):
        for df,name in zip([self.df_A,self.df_B,self.df_C,self.df_D],self.names):
            print("Building map for", name )
            outputname=outputdir+"/"+name+".hdf5"
            diro = os.path.dirname(outputname)
            os.makedirs(diro,exist_ok=True)
            print("output will be at ", outputname,"starting on mapBuilder Constructor")
            mapB = mb.MapBuilder(df,df.shape[0])
            print("run()")
            mapB.run()
            print("build hdf5")
            mapB.build(binning_scheme=input_binning_scheme, file_name=outputname, use_weights=False)
            print("Done on ",name)

    def loadMasterMap(self,dir):
            print("loading master maps from ", dir)
            self.map_A = h5py.File(dir+"/"+self.names[0]+".hdf5", "r")
            self.map_B = h5py.File(dir+"/"+self.names[1]+".hdf5", "r")
            self.map_C = h5py.File(dir+"/"+self.names[2]+".hdf5", "r")
            self.map_D = h5py.File(dir+"/"+self.names[3]+".hdf5", "r")
            print("Now available to helper as self.map_X")

    def reweight(self,targetMap,weightName):
        print("Starting to reweight master map to target, weight name ",weightName)
        start = time.time()
        self.masterFinal[weightName] = self.masterFinal.apply(lambda x: getMasterWeight(self.map_A,self.map_B,self.map_C,self.map_D, targetMap, np.array([x['true_energy_e_minus'], x['true_energy_sum'], x['true_delta_theta'], x['true_pz_p_e_plus'],x['true_pz_p_e_minus'], x['true_phi_e_minus']]), self.BP_corr) , axis=1);
        end = time.time()
        print("Apply the map took ", end - start)
        print('Some (possibly useful) info')
        print("Outside Binning: ",self.masterFinal[self.masterFinal[weightName]==-999][weightName].count())
        print("Div by Zero: ",self.masterFinal[self.masterFinal[weightName]==-1e-10][weightName].count())
        print("Zero: ",self.masterFinal[self.masterFinal[weightName]==0][weightName].count())
        print("Valid: ",self.masterFinal[self.masterFinal[weightName]>0][weightName].count())
        print("Mean Non-zero Weight: ",self.masterFinal[self.masterFinal[weightName]>0][weightName].mean())
        print("Mean Weight: ",self.masterFinal[weightName].mean())
        return self.masterFinal














        
    
      


    
    





  
          


