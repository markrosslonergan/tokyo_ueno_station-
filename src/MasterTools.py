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
        masterFinal = pd.concat([gLEE_dfA_fin,gLEE_dfB_fin,gLEE_dfC_fin,gLEE_dfD_fin])

    def loadMasterOrig(self):
        gLEE_dfD_vert = loadgLEE_Bare(self.datadir+"/vertex_DarkNu_Run1_Keystone_BenchD_noHF_v50.0.root")
        gLEE_dfA_vert = loadgLEE_Bare(self.datadir+"/vertex_DarkNu_Run1_Keystone_BenchAp_noHF_v50.0.root")
        gLEE_dfC_vert = loadgLEE_Bare(self.datadir+"/vertex_DarkNu_Run1_Keystone_BenchC_noHF_v50.0.root")
        gLEE_dfB_vert = loadgLEE_Bare(self.datadir+"/vertex_DarkNu_Run1_Keystone_BenchBp_noHF_v50.0.root")
        self.masterOrig = pd.concat([gLEE_dfA_vert,gLEE_dfB_vert,gLEE_dfC_vert,gLEE_dfD_vert])
         
    
    def loadMasterDFs(self):
        self.df_A = pd.read_pickle("/home/mark/work/DarkNu_Codebase_Nov2021/workin_dir/Map_Develop/microboone_study/FilteredMapGeneration/data/microboone_active_tpc_benchmark/3plus2/m5_0.15_m4_0_mzprime_0.03_dirac/pandas_df.pckl")
        self.df_B = pd.read_pickle("/home/mark/work/DarkNu_Codebase_Nov2021/workin_dir/Map_Develop/microboone_study/FilteredMapGeneration/data/microboone_active_tpc_benchmark/3plus2/m5_0.15_m4_0_mzprime_1.25_dirac/pandas_df.pckl")
        self.df_C = pd.read_pickle("/home/mark/work/DarkNu_Codebase_Nov2021/workin_dir/Map_Develop/microboone_study/FilteredMapGeneration/data/microboone_active_tpc_benchmark/3plus2/m5_0.15_m4_0.107_mzprime_0.03_dirac/pandas_df.pckl")
        self.df_D = pd.read_pickle("/home/mark/work/DarkNu_Codebase_Nov2021/workin_dir/Map_Develop/microboone_study/FilteredMapGeneration/data/microboone_active_tpc_benchmark/3plus2/m5_0.15_m4_0.107_mzprime_1.25_dirac/pandas_df.pckl")        

    def buildMasterMap(self,input_binning_scheme,outputdir):
        for df,name in zip([self.df_A,self.df_B,self.df_C,self.df_D],self.names):
            print("Building map for", name )
            outputname=outputdir+"/"+name+".hdf5"
            diro = os.path.dirname(outputname)
            os.makedirs(diro,exist_ok=True)
            print("output will be at", outputname)
            mapB = mb.MapBuilder(df,df.shape[0])
            mapB.run()
            mapB.build(binning_scheme=input_binning_scheme, file_name=outputname, use_weights=False)
            print("Done on ",name)

    def loadMasterMap(self,dir):
            print("loading master maps from ", dir)
            self.map_A = h5py.File(dir+"/"+self.name[0]+".hdf5", "r")
            self.map_B = h5py.File(dir+"/"+self.name[1]+".hdf5", "r")
            self.map_C = h5py.File(dir+"/"+self.name[2]+".hdf5", "r")
            self.map_D = h5py.File(dir+"/"+self.name[3]+".hdf5", "r")
            print("Now available to helper as self.map_X")






        
    
      


    
    





  
          


