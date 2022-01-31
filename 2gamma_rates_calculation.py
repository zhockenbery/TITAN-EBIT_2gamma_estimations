#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:26:00 2022

@author: zachary_hockenbery
"""

import os
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import integrate
import csv
#import pandas as pd
import sys
#import pyne;
import argparse;
import configparser;
# import time
from datetime import *

from pyne import nucname;
from pyne.material import Material;
from pyne.transmute.chainsolve import Transmuter;
from pyne import data;

#%%




class trap_contents:
    def __init__(self,
                 cbsim3_folder='',
                 filename_out='',
                 time_in_trap_per_cycle=0,
                 injection_frequency=0,
                 total_injections=0,
                 primary_beam='',
                 bunch_size=0,
                 isac_rate=0,
                 tspan=0,
                 beta_population=0,
                 stacked_population=0,
                 cbsim3_population=0,
                 BR_to_isomer=0,
                 isomer_rates=0,
                 halflives_2gamma=0,
                 decay_rates_2gamma=0,
                 isomer_population=0,
                 isomer_population_corrected=0,
                 crude_estimate_2gamma = 0,
                 folder_out = '',
                 ):
        self.cbsim3_folder = cbsim3_folder
        self.filename_out = filename_out
        self.time_in_trap_per_cycle = time_in_trap_per_cycle
        
        self.injection_frequency = injection_frequency
        self.total_injections    = total_injections
        self.primary_beam        = primary_beam
        self.bunch_size          = bunch_size
        self.isac_rate           = isac_rate
        
        # these aren't inputs, I generate them during simulation
        self.tspan           = tspan
        self.beta_population = beta_population # processed out of PyNE Material()
        self.stacked_population = stacked_population
        self.cbsim3_population = cbsim3_population
        self.BR_to_isomer = BR_to_isomer
        self.isomer_rates = isomer_rates
        self.halflives_2gamma = halflives_2gamma
        self.decay_rates_2gamma = decay_rates_2gamma
        self.isomer_population = isomer_population
        self.isomer_population_corrected = isomer_population_corrected
        self.crude_estimate_2gamma = crude_estimate_2gamma
        self.folder_out = folder_out
        
        return
    
    
    def importCBSIM3(self):
        """
        Imports the data file from CBSIM. Performs a linear interpolation
        and returns the function. The function can be called as f(x)
        and returns the value up until the end of the data points.
        """
        self.cbsim3_population = {"Y98":[],
                                  "Nb98":[],
                                  }

        fids = os.listdir(self.cbsim3_folder)
        fids = [i for i in fids if ".DAT" in i]
        for fid in fids:
            with open(self.cbsim3_folder+os.sep+fid, 'r') as fopen:
                data = fopen.readlines()
            headerStart = int(data.index(" &VARIABLES\n"))
            headerEnd = int(data.index(" &END\n"))
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
                
                print("\n==== importing CBSIM3 file ====")
                print("imported "+fid+", final ts = %s seconds"%timestamps[-1] )
                print("Current density is %s A/cm^2"%headerDict["DENS"])
                print("e-beam energy is %s eV"%headerDict["ENERGY"])
                print("--------\n")
            else:
                print("Failure to import cbsim3 file")
                sys.exit()
            self.cbsim3_population[fid.split("_")[0]] = interp1d([0]+list(timestamps), [0]+list([*zip(*population)][-1]) )
            

            plt.figure(figsize=(12,8))
            plt.plot(timestamps, list([*zip(*population)][-1]))
            plt.title("Population of bare ions of %s"%fid.split("_")[0])
            plt.xlabel("time [s]")
            plt.ylabel("population")
            plt.xscale("log")
            plt.savefig(self.folder_out+os.sep+self.filename_out+"_CBSIM3_%s"%fid.split("_")[0], dpi=100)


        if timestamps[-1] < self.tspan[-1]:
            print("time duration of CBSIM3 file is less than EBIT trapping time!")
            sys.exit()

        for key in self.cbsim3_population:
            self.beta_population[key] = np.multiply(self.beta_population[key], self.cbsim3_population[key](self.tspan) )
        
        return

    
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
        inp = Material(initial_pop, mass=1.0);
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
    

    def isomer_production_rate(self):
        print("Calculating isomer production rate. Estimated half-life of CX and RR in EBIT is around 590 ms. So we assume worst case and that 25 percent of bare 0+ isomers are lost to IC")

        self.isomer_rates = {"Y98": [],
                             "Nb98": [],
                             }

        for key in self.BR_to_isomer:
            self.isomer_rates[key] = np.multiply(self.stacked_population[key], 0.75*data.decay_const(key)*self.BR_to_isomer[key] )

        return

    def calculate_isomer_population(self):
        """ To calcualte this, we need the isomer_rates array. This tells use the
        rate at which the 0+ isomers are produced as a function of time.

        We make an assumption here that the rate of 0+ production is not affected
        by the 2gamma decay. I.e. it happens so slowly that we only siphon off a little
        bit of the main beta decay path.
        """

        # for key in self.isomer_rates:
        #     f = interp1d(self.tspan, self.isomer_rates[key])
        #     val = integrate.quad(f, 0, self.tspan[-1])[0]
        #     print("total bare 0+ isomers of %s produced: %s"%(key, val))

        # holds the isomer population (without losses to 2gamma)
        self.isomer_population = {"Y98": np.zeros(len(self.tspan)-1),
                                   "Nb98": np.zeros(len(self.tspan)-1),
                                   }
        # hols the number of 2gamma decays that occur per
        self.decay_rates_2gamma = {"Zr98": np.zeros(len(self.tspan)-1),
                                   "Mo98": np.zeros(len(self.tspan)-1),
                                   }
        self.isomer_losses = {"Zr98": np.zeros(len(self.tspan)),
                              "Mo98": np.zeros(len(self.tspan)),
                              }

        self.crude_estimate_2gamma = {"Zr98": 0.0,
                                      "Mo98": 0.0,
                                      }

        dt_array = np.diff(self.tspan)
        # isomer rates only contains "Y98" and "Nb98"
        for key in self.isomer_rates:
            f = interp1d(self.tspan, self.isomer_rates[key])
            key_children = [*map(nucname.name, [*data.decay_children(key)])]

            # determines the correct key to store data in self.decay_rates_2gamma
            key_2gamma = [*set(key_children) & set(self.decay_rates_2gamma.keys())][0]

            # Omit the first value here:
            for idx, val in enumerate(self.tspan[1:]):
                self.isomer_population[key][idx] = integrate.quad(f, 0, val, limit=len(self.tspan))[0]

            self.crude_estimate_2gamma[key_2gamma] = integrate.quad(f, 0, self.tspan[-1], limit=len(self.tspan))[0]
            
            for idx, val in enumerate(self.isomer_population[key]):
                # update the population based on what was lost before
                val = val - self.isomer_losses[key_2gamma][idx]

                # calculate how many decays from current value
                self.decay_rates_2gamma[key_2gamma][idx] = np.log(2)/self.halflives_2gamma[key_2gamma]* val * dt_array[idx]
                
                self.isomer_losses[key_2gamma][idx+1] = self.isomer_losses[key_2gamma][idx] + self.decay_rates_2gamma[key_2gamma][idx]

        print("\n==== Calculated 2gamma decay rates ====\n")
        for key in self.decay_rates_2gamma:
            print("2gamma decays from $0+$ %s isomer: %s in %s seconds"%(key, sum(self.decay_rates_2gamma[key]), self.time_in_trap_per_cycle))
            print("  --> Compare to the crude estimate: %s per trap cycle"%self.crude_estimate_2gamma[key])
            print("This is after the 25 percent IC loss, so that's %s decays/sec\n"%(sum(self.decay_rates_2gamma[key])/self.time_in_trap_per_cycle))
            

        return

    def write_report(self):
        """ Write a report to a txt file for saving later
        I also need to write the data to a csv file!

        self.cbsim3_folder = cbsim3_folder
        self.filename_out = filename_out
        self.time_in_trap_per_cycle = time_in_trap_per_cycle
        
        self.injection_frequency = injection_frequency
        self.total_injections    = total_injections
        self.primary_beam        = primary_beam
        self.bunch_size          = bunch_size
        self.isac_rate           = isac_rate
        
        # these aren't inputs, I generate them during simulation
        self.tspan           = tspan
        self.beta_population = beta_population # processed out of PyNE Material()
        self.stacked_population = stacked_population
        self.cbsim3_population = cbsim3_population
        self.BR_to_isomer = BR_to_isomer
        self.isomer_rates = isomer_rates
        self.halflives_2gamma = halflives_2gamma
        self.decay_rates_2gamma = decay_rates_2gamma
        self.isomer_population = isomer_population
        self.isomer_population_corrected = isomer_population_corrected
        self.crude_estimate_2gamma = crude_estimate_2gamma
        with open(self.folder_out+os.sep+self.filename_out+"_report.txt", 'w') as fopen:
        """
            csvwriter = csv.writer(fopen, delimiter=',', quoting=csv.QUOTE_NONE, escapechar="%")
            csvwriter.writerow(["==== Simulation report ===="])

            csvwriter.writerow(["  ___injection___"])
            csvwriter.writerow(['primary beam: %s, ISAC rate %s /s'%(self.primary_beam, self.isac_rate)])
            csvwriter.writerow(["injection frequency: %s s"%self.injection_frequency])
            csvwriter.writerow(["number of injections: %s, time elapsed during injections: %s s"%(self.injection_frequency, self.total_injections/self.injection_frequency)])
            csvwriter.writerow(["EBIT bunch size (~6 percent ISAC-to-EBIT efficiency: %s "%self.bunch_size])
            csvwriter.writerow(["duration of one trapping cycle %s s"%self.time_in_trap_per_cycle])
            

            csvwriter.writerow(["  ___final results___"])

            csvwriter.writerow(["END"])



        return

    def write_dictionary(self, data, name):
        """ This is to write out the various dictionaries that contain our population data

        It should be EASY to read back in so I can write a separate script for plotting multiple
        results.
        """
        with open(self.folder_out+os.sep+self.filename_out+"_"+name+".txt", 'w') as fopen:
            fieldnames = data.keys()
            # how to get tspan in here?
            csvwriter = csv.DictWriter(fopen, fieldnames)
            csvwriter.writeheader()
            csvwriter.writerows([data])
            
    

def save_plot(trap_contents,
                filename_out,
                tspan,
                population_data,
                yscale="linear",
                xscale="linear",
                xlabel="population",
                title="default title",
                selector=[],
                ):

    print("Saving plot")
    if selector==[]:
        keys = population_data.keys()
    else:
        keys = selector
    plt.figure(figsize=(12,9))
    for key in keys:
        plt.plot(tspan, population_data[key], label=key)
        
    plt.xlabel("time [seconds]")
    plt.ylabel(xlabel)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.title(title)
    plt.legend()
    plt.savefig(trap_contents.folder_out+os.sep+filename_out, dpi=100)
    
    return

def interpolate_cbsim(tspan, population):

    # simple linear is fine for this monotonic charge breeding curve
    f = interp1d(tspan, population)


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

    save_plot(my_trap_contents,
              my_trap_contents.filename_out,
              my_trap_contents.tspan,
              my_trap_contents.beta_population,
              title="Single injection population",
              yscale="log",
              xscale="log",
              )

    # Convert to bare ions
    # this plot will only have 98Y and 98Nb (parents of isomers)
    # this function modifies self.beta_population, so its now
    # the single bunch population of BARE ions
    my_trap_contents.importCBSIM3()

    save_plot(my_trap_contents,
              my_trap_contents.filename_out+"_bare",
              my_trap_contents.tspan,
              my_trap_contents.beta_population,
              title="Single injection population, bare",
              yscale="log",
              xscale="log",
              selector=["Y98","Nb98"],
              )
    
    # Stack the bare ions!
    my_trap_contents.stack_beam()
    
    save_plot(my_trap_contents,
              my_trap_contents.filename_out+"_stacked_bare",
              my_trap_contents.tspan,
              my_trap_contents.stacked_population,
              title="Population stacked at %s Hz for %s s duration, bare"%(my_trap_contents.injection_frequency, my_trap_contents.total_injections/my_trap_contents.injection_frequency),
              yscale="log",
              xscale="log",
              selector=["Y98","Nb98"],
              )

    # Multiply by lambda_beta and BR to 0+
    my_trap_contents.isomer_production_rate()


    save_plot(my_trap_contents,
              my_trap_contents.filename_out+"_isomer_rates",
              my_trap_contents.tspan,
              my_trap_contents.isomer_rates,
              # yscale="log",
              xscale="log",
              xlabel="Rate [1/s]",
              title=r"$0_2^+$ isomer production rates, stacked injection")


    
    # print("tspan:")
    # print("["+", ".join(map(str, my_trap_contents.tspan))+"]" )
    # print("isomer rate:")
    # print("["+", ".join(map(str, my_trap_contents.isomer_rates["Y98"]) )+"]")

    my_trap_contents.calculate_isomer_population()

    my_trap_contents.write_report()

    my_trap_contents.write_dictionary(my_trap_contents.beta_population, "beta_population")

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


        #input stuff
        my_trap_contents.cbsim3_folder =              getConfigEntry(config, 'input', 'cbsim3_folder', reqd=True, remove_spaces=True)
        # output stuff
        my_trap_contents.filename_out =               getConfigEntry(config, 'output', 'filename_out', reqd=True, remove_spaces=True)
        # injection stuff
        my_trap_contents.primary_beam =               getConfigEntry(config, 'injection', 'primary_beam', reqd=True, remove_spaces=True)
        my_trap_contents.isac_rate =            float(getConfigEntry(config, 'injection', 'isac_rate', reqd=True, remove_spaces=True))
        my_trap_contents.injection_frequency =    int(getConfigEntry(config, 'injection', 'injection_frequency', reqd=True, remove_spaces=True))
        # trapping stuff
        my_trap_contents.total_injections =       int(getConfigEntry(config, 'trapping', 'total_injections', reqd=True, remove_spaces=True))
        my_trap_contents.time_in_trap_per_cycle = int(getConfigEntry(config, 'trapping', 'time_in_trap_per_cycle', reqd=True, remove_spaces=True))
        
        print("You have entered an ISAC rate of %s /s of %s"%(my_trap_contents.isac_rate,my_trap_contents.primary_beam))
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

    # some other pre-determined data
    my_trap_contents.BR_to_isomer = {"Y98": 1.5e-1,
                                    "Nb98": 2.6e-1,
                                    }
    # extrapolated from curve fit, seconds
    my_trap_contents.halflives_2gamma = {"Zr98": 0.054,
                                         "Mo98": 0.154,
                                         }

    # for a unique timestamp folder name
    my_trap_contents.folder_out = my_trap_contents.filename_out+"_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    os.mkdir(my_trap_contents.folder_out)
    # start_time = time.time()
    runSimulation(my_trap_contents)
    # stop_time = time.time()
    # print("Simulation ran in %s seconds (using time.time(), so it's not perfect)"%(stop_time-start_time))
    
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