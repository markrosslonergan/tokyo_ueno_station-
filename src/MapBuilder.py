#!/usr/bin/env python3
import numpy as np
import h5py
import argparse
import shutil
import os
import os.path
from DarkNews import GenLauncher, AssignmentParser, Cfourvec as Cfv

import time

#import analysis_decay as a_d


class MapBuilder:
    """
        Builds the map from a DarkNews run.
        It first computes the different observables, stacking them into a numpy.ndarray self._data.
        That array has shape (len(dataframe), 6), because there are 6 observables.
        It is possible to build the actual map (aka the histogram) passing a binning scheme to
        the method build.
        The order of the observables in the self._data array is:
          - Energy e- (energy_e_minus)
          - Total energy of e+e- pair (energy_sum)
          - Delta theta (delta_theta): separation angle of e+e- pair
          - pz/p for e+ (pz_p_e_plus)
          - pz/p for e- (pz_p_e_minus)
          - azimutal angle phi for e- (phi_e_minus): from positive x-axis direction counterclockwise
    """

    def __init__(self, input_df, neval):
        self._dataframe = input_df
        self._neval = neval

    def run(self):
        
        start = time.time()
        #Not run from scratch, use input dt
        #self._dataframe = self.run_obj.run(overwrite_path=run_path, loglevel="ERROR")
        self.weights = self._dataframe["w_event_rate"].values
        #self.ctaus = self._dataframe.attrs["N5_ctau0"] 
        
        mid = time.time()
        #print("Time Gen : ",mid - start)

        #print("Starting to decay: Sum(w_event_rate) ",self._dataframe["w_event_rate"].sum())
        #print("Decay length is : ",self.ctaus," cm")
        #self._dataframe = a_d.select_muB_decay_prob(self._dataframe, l_decay_proper_cm = self.ctaus)
        #self._dataframe = a_d.select_muB_decay_prob(self._dataframe)
        #print("Finishing decay: Sum(w_event_rate) is ",self._dataframe["w_event_rate"].sum())

        end = time.time()
        #print("Time Decay: ",end - mid)

        # observables
        self.energy_e_plus = self._get_energy("P_decay_ell_plus")
        self.energy_e_minus = self._get_energy("P_decay_ell_minus")
        self.energy_sum = self.energy_e_plus + self.energy_e_minus
        self.delta_theta = self._get_opening_angle()
        self.weights = self._get_p_4v("w_event_rate")
        self.pos_decay_z = self._get_pos_decay_z("pos_decay")
        _, _, self.pz_p_e_plus = self._get_pi_over_p("P_decay_ell_plus")
        _, _, self.pz_p_e_minus = self._get_pi_over_p("P_decay_ell_minus")
        _, _, self.phi_e_minus = self._get_spherical_coords("P_decay_ell_minus")
        # data
        self._data = np.vstack((
            self.energy_e_minus,
            self.energy_sum,
            self.delta_theta,
            self.pz_p_e_plus,
            self.pz_p_e_minus,
            self.phi_e_minus,
            self.pos_decay_z
        )).T
    def _get_pos_decay_z(self, entry):
        return self._dataframe[(entry, "3")].values
   
    
    def _get_energy(self, entry):
        return self._dataframe[(entry, "0")].values

    def _get_p_4v(self, entry):
        return self._dataframe[entry].values
        p_4v = self._get_p_4v(entry)
        p_3v_norm = Cfv.get_3vec_norm(p_4v)
        cos_theta = Cfv.get_cosTheta(p_4v)
        px = p_4v[:,1]
        py = p_4v[:,2]
        phi = np.arctan2(py, px) / np.pi * 180
        return p_3v_norm, cos_theta, phi
 
    def _get_spherical_coords(self, entry):
        """
            Returns spherical coordinates p, cos_theta, phi.
        """
        p_4v = self._get_p_4v(entry)
        p_3v_norm = Cfv.get_3vec_norm(p_4v)
        cos_theta = Cfv.get_cosTheta(p_4v)
        px = p_4v[:,1]
        py = p_4v[:,2]
        phi = np.arctan2(py, px) / np.pi * 180
        return p_3v_norm, cos_theta, phi


    def _get_opening_angle(self):
        p1 = self._get_p_4v("P_decay_ell_plus")
        p2 = self._get_p_4v("P_decay_ell_minus")
        return np.arccos(Cfv.get_cos_opening_angle(p1, p2)) / np.pi * 180.

    def _get_pi_over_p(self, entry):
        p_4v = self._get_p_4v(entry)
        return Cfv.get_3direction(p_4v).T

        the_map, the_bins = np.histogramdd(sample=self._data, bins=binning_scheme, weights=self.weights)
    def build(self, binning_scheme, file_name, use_weights=True):
        """
            Create the map and saves it to the file file_name in hdf5 format
            (preferred extension is hdf5).
            It also stores the version of this script as well as the order of
            the observables bins (the same that it should be used in the 
            binning scheme argument) and the arrays with the bin edges.
            In summary, the following keys:
              - 'version': accesses the version of this script;
              - 'observables': access the array of the observables in order
              - $observable_name: as written in the 'observables' array, accesses
                the bin edges for that observable, e.g. 'energy_sum' key is the
                array containing the bin edges for the observable 'Total energy of e+e- pair'
        """
        if use_weights:
            print("Including Weights")
            the_map, the_bins = np.histogramdd(sample=self._data, bins=binning_scheme, weights=self.weights)
        else :
            the_map, the_bins = np.histogramdd(sample=self._data, bins=binning_scheme)
        
        with h5py.File(file_name, "w") as h5w:
            h5w.attrs["version"] = "MapBuilder_FromDF_v0.1.0"
            #h5w.attrs["GenLauncher_param_file"] = self._param_file
            h5w.attrs["GenLauncher_neval"] = self._neval
            h5w.attrs["observables"] = ["energy_e_minus", "energy_sum", "delta_theta", "pz_p_e_plus", "pz_p_e_minus", "phi_e_minus","pos_decay_z"]
            for i, attr in enumerate(h5w.attrs["observables"]):
                h5w.attrs[attr] = the_bins[i]
            h5w.create_dataset("map", the_map.shape, dtype="f", data=the_map)
            h5w.attrs["Summed_EventRate"] = self._dataframe["w_event_rate"].sum()
            #h5w.attrs["N5_ctau0"] = self._dataframe.attrs["N5_ctau0"]
          


