#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:26:00 2022

@author: zachary_hockenbery
"""

import os
import numpy as np 
import matplotlib.pyplot as plt
#import csv
#import pandas as pd
import sys
#import pyne;
import argparse;
import configparser;

from pyne import nucname;
from pyne.material import Material;
from pyne.transmute.chainsolve import Transmuter;
from pyne import data;

#%%

class trap_contents:
    def __init__(self,
                 filename_out='',
                 time_in_trap_per_cycle=0,
                 trap_cycles=0,
                 injection_frequency=0,
                 total_injections=0,
                 primary_beam='',
                 bunch_size=0,
                 isac_rate=0,
                 tspan=0,
                 tspan_corrected=0,
                 beta_population=0,
                 stacked_population=0
                 ):
        self.filename_out = filename_out
        self.time_in_trap_per_cycle = time_in_trap_per_cycle
        self.trap_cycles = trap_cycles
        
        self.injection_frequency = injection_frequency
        self.total_injections    = total_injections
        self.primary_beam        = primary_beam
        self.bunch_size          = bunch_size
        self.isac_rate           = isac_rate
        
        # these aren't inputs, I generate them during simulation
        self.tspan           = tspan
        self.tspan_corrected = tspan_corrected
        self.beta_population = beta_population # processed out of PyNE Material()
        self.stacked_population = stacked_population
        
        return
    
    
    def importCBSIM3(self, inputFile):    
        
        with open(inputFile, 'r') as fopen:
            data = fopen.readlines()

        headerStart = int(data.index(" &VARIABLES\n"))
        headerEnd = int(data.index(" &END\n"))
        print("header ends at %s"%headerEnd)
        headerDict = {}
        for dval in data[headerStart+1:headerEnd]:
            headerDict[dval.split()[0]] = dval.split("=")[-1]
            
        Z = int(headerDict["NORDER"][:-2])
        if any(['Charge abundance at the steps' in i for i in data]):
            trth = ['Charge abundance at the steps' in i for i in data]
            idc = int([*filter(lambda i: trth[i], range(len(trth)))][0] )
            # idc + 5 gives first time step, +5 for each subsequent time step
            data = data[idc+5:] # chop off the unwanted stuff.
            
            # data chuncks have variable length (num charge states), use blank line to find chunks
            stepdata = [[] for i in range(data.count(' \n')+1) ]
            skips = [' \n' in i for i in data]
            idcs = [*filter(lambda i: skips[i], range(len(skips)))]
            idcs = [0]+[i+1 for i in idcs]
            timestamps = np.zeros(len(stepdata)-1)
            population = np.zeros((len(stepdata)-1,Z+1))
            for idx, d in enumerate(stepdata):
                if idx == len(stepdata)-1:
                    stepdata[idx] = data[idcs[idx]:]
                else:
                    stepdata[idx] = data[idcs[idx]:idcs[idx+1]-1]
            # first index of sub-list is time step. All others contain charge state and abundance.
            # technically, we only want the final value, but take -2 because -1 is just a timestamp alone (??)
            for idx, d in enumerate(stepdata[:-1]):
                timestamps[idx] = float(d[0])
                tempdata = [*map(lambda x: [int(x.split()[0]),float(x.split()[1])] ,d[1:])]
                for val in tempdata:
                    population[idx][val[0]] = val[1]
            
            print("========== importing CBSIM3 file ============")
            print("imported "+inputFile+", final ts = %s seconds"%timestamps[-1] )
            print("Current density is %s"%headerDict["DENS"])
            print("Ebeam energy is %s"%headerDict["ENERGY"])
            print("\n")
        else:
            "Error, wrong data file or something..."
            return timestamps, population
    
    def generate_beta_population(self):
        """ Uses PyNE to generate the population due to beta decay chain
        Gives back in an array form instead of the PyNE Material() class.
        """
        
        # generate a time array with dt = 1/injection_frequency
        # required for the ion stacking
        num_points = self.injection_frequency * self.time_in_trap_per_cycle
        self.tspan = np.linspace(0, self.time_in_trap_per_cycle, num_points)
        
        # use initial population assigned from config file
        initial_pop = {self.primary_beam: 1.0}
        
        population = [[] for i in range(num_points)]
        inp = Material(initial_pop, mass=1.0)
        population[0] = inp
        
        for idx,tval in enumerate(self.tspan[1:]):
            population[idx+1] = probeDecayChain(initial_pop, tval)
        
        
        population_array = {"Rb98": [[] for i in range(num_points)],
                            "Sr98": [[] for i in range(num_points)],
                            "Y98": [[] for i in range(num_points)],
                            "Zr98": [[] for i in range(num_points)],
                            "Nb98": [[] for i in range(num_points)],
                            "Mo98": [[] for i in range(num_points)],}
        
        for idx, val in enumerate(population_array):
            
            for idt, pval in enumerate(population):
                try:
                    population_array[val][idt] = pval[val]*self.bunch_size
                except KeyError:
                    population_array[val][idt] = 0.0
        # give population to class instance
        self.beta_population = population_array
        return
    
    def stack_beam(self):

        num_points = self.injection_frequency * self.time_in_trap_per_cycle
        
        self.stacked_population = {"Rb98": [[] for i in range(num_points)],
                                   "Sr98": [[] for i in range(num_points)],
                                   "Y98": [[] for i in range(num_points)],
                                   "Zr98": [[] for i in range(num_points)],
                                   "Nb98": [[] for i in range(num_points)],
                                   "Mo98": [[] for i in range(num_points)],}
        
        for idx, val in enumerate(self.stacked_population):
            self.stacked_population[val] = add_shifted_array(self.beta_population[val], self.total_injections)
        
        print("Stacking beam... %s Hz injection for %s seconds"%(self.injection_frequency, self.total_injections/self.injection_frequency))
        print("After stacking, total population is %s ions"%sum([self.stacked_population[i][-1] for i in self.stacked_population]))
        print("Keep in mind that EBIT trap capacity is 1e7 ions.")
        return 
    

def save_plot(filename_out, tspan, population_data, title="default title"):
    print("Saving plot")
    plt.figure(figsize=(12,9))
    for key in population_data.keys():
        plt.plot(tspan, population_data[key], label=key)
        
    plt.xlabel("time [seconds]")
    plt.ylabel("population")
    plt.title(title)
    plt.legend()
    plt.savefig(filename_out, dpi=100)
    
    return

def add_shifted_array(array, num_shifts):
    """ This actually performs the beam stacking"""
    array0 = array[:]
    
    if num_shifts > len(array):
        sys.exit("can't have more stacks than the total trap time!")
    
    for num in range(1, num_shifts):
        array = array + np.insert(array0, [0]*num,0)[:-num]
    
    return array

def probeDecayChain(material, timestamp):
    # How to use this: input a population in the form
    #     population = {"72Ga", 1.0}
    # the second number is the population fraction, if multiple species are used
    # Input a delta_t, which steps it forward in time and tells us what the new
    # population is.
    tm  = Transmuter(tol=1e-10)
    decay_chain = tm.transmute(material, t=timestamp)
    
    return decay_chain

def runSimulation(my_trap_contents):
    print("Running simulation!")
    
    # make this once so we can generate the time array
    num_points = my_trap_contents.injection_frequency * my_trap_contents.time_in_trap_per_cycle
    
    my_trap_contents.tspan = np.linspace(0,
                                         my_trap_contents.time_in_trap_per_cycle,
                                         num_points)
    
    # doesnt need inputs, already has self, doesn't need output, already has self
    my_trap_contents.generate_beta_population()
    
    my_trap_contents.stack_beam()
    
    save_plot(my_trap_contents.filename_out,
              my_trap_contents.tspan,
              my_trap_contents.beta_population,
              title="Single injection population")
    
    save_plot(my_trap_contents.filename_out+"_stacked",
              my_trap_contents.tspan,
              my_trap_contents.stacked_population,
              title="Population stacked at %s Hz for %s s duration"%(my_trap_contents.injection_frequency, my_trap_contents.total_injections/my_trap_contents.injection_frequency) )
    
    
    
    return

def getConfigEntry(config, heading, item, reqd=False, remove_spaces=True, default_val=''):
    #  Just a helper function to process config file lines, strip out white spaces and check if requred etc.
    if config.has_option(heading, item):
        if remove_spaces:
            config_item = config.get(heading, item).replace(" ", "")
        else:
            config_item = config.get(heading, item)
    elif reqd:
        print("The required config file setting \'%s\' under [%s] is missing") % (item, heading)
        sys.exit(1)
    else:
        config_item = default_val
    return config_item


def processConfigFile(configFileName):
    # read in the entire config file here and push it into the trap_contents class
    config = configparser.RawConfigParser()
    
    
    if os.path.exists(configFileName):
        config.read(configFileName)
        my_trap_contents = trap_contents()
        
        # output stuff
        my_trap_contents.filename_out           = getConfigEntry(config, 'output', 'filename_out', reqd=True, remove_spaces=True)
        # injection stuff
        my_trap_contents.primary_beam           = getConfigEntry(config, 'injection', 'primary_beam', reqd=True, remove_spaces=True)
        my_trap_contents.isac_rate             = float(getConfigEntry(config, 'injection', 'isac_rate', reqd=True, remove_spaces=True))
        my_trap_contents.injection_frequency    = int(getConfigEntry(config, 'injection', 'injection_frequency', reqd=True, remove_spaces=True))
        # trapping stuff
        my_trap_contents.total_injections       = int(getConfigEntry(config, 'trapping', 'total_injections', reqd=True, remove_spaces=True))
        my_trap_contents.time_in_trap_per_cycle = int(getConfigEntry(config, 'trapping', 'time_in_trap_per_cycle', reqd=True, remove_spaces=True))
        
        print("You have entered an ISAC rate of %s /s "%my_trap_contents.isac_rate)
        print("ISAC to TITAN-RFQ transport efficiency: 0.85")
        print("At %s Hz RFQ operation, this gives a RFQ bunch size of %s /s"%(my_trap_contents.injection_frequency, my_trap_contents.isac_rate*0.85/my_trap_contents.injection_frequency))
        print("The following TITAN transport efficiencies are assumed:")
        print("    through TITAN-RFQ: 0.10")
        print("    RFQ to TITAN-EBIT: 0.70")
        my_trap_contents.bunch_size = 0.85*0.10*0.70*my_trap_contents.isac_rate/my_trap_contents.injection_frequency
        print("So actual rate of injection into EBIT is %s PER BUNCH... \n" %my_trap_contents.bunch_size)
        
    else:
        print("Config file doesn't exist: %s" %configFileName)
        sys.exit(1)
    
    runSimulation(my_trap_contents)
    
def main():
    print("\n======== TITAN-EBIT 2-gamma count rate simulator ========\n\n")
    print("                 Author: Z. Hockenbery")
    print("                    Date: Jan 2022")
    print("\n=========================================================\n")
    parser = argparse.ArgumentParser(description="2gamma estimation simulation")
    parser.add_argument('--configFile', dest='configFile', required=True,
                        help="specify the path to the config file")
    args, unknown = parser.parse_known_args()
    
    processConfigFile(args.configFile)
    
if __name__ == "__main__":
    main()