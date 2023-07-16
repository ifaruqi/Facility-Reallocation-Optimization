#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:15:54 2023

@author: ismailfaruqi
"""

import numpy as np
import pandas as pd
import pulp as pl
import random
import time


##############################################################################
########################## Data Preparation ##################################
##############################################################################

# Load required data
file_name='facilitiesGB.xlsx'
facility = pd.read_excel(file_name, "Facilities", index_col=1)
customer = pd.read_excel(file_name, "Local Authorities", index_col=0)
distance = pd.read_excel(file_name, "Distance Matrix", index_col=0)/60

# Define demand and supply
customer["demand"] = round(customer["Population (2021)"] * 0.001)
facility["supply"] = round(facility["Capacity (1000 visits)"]*1000)

#customer = customer.sort_values("demand", ascending=False)


capacity = facility['supply'].copy()
demand = customer['demand'].copy()


##############################################################################
################################ Functions ###################################
##############################################################################

# Function to check total allocation.
# Result need to be 374 otherwise single-sourcing constraint is violated or
# Customer is not allocated
def calculate_total_allocation(allocation):
    total_allocation = 0
    for i in allocation.index:
        for j in allocation.columns:
            if allocation.loc[i,j] ==1:
                print("customer ", i, "allocated to facility ", j) # for troubleshoot only
                total_allocation += allocation.loc[i,j]
    return total_allocation

# Function to compute total cost for each allocation
def calculate_cost(allocation):
    cost = 0
    for i in allocation.index:
        for j in allocation.columns:
            if allocation.loc[i, j] == 1:
                cost += customer.loc[i, 'demand'] * distance.loc[i,j]
    return(cost)


# Relaxation 
#facility["supply"] = facility["supply"] * 3
#capacity = capacity * 3


##############################################################################
################### Sequential Constructive Heuristics #######################
##############################################################################

# Calculate distances between facilities and customers
dist_mat = distance.loc[customer.index, facility.index].values

# Rank facilities by distance to each customer
ranked_facilities = np.argsort(dist_mat, axis=1)

t_SE = time.time()

# Allocate customers to facilities using the constructive heuristic algorithm
def allocate_customers(dist_mat, demand, supply, ranked_facilities):
    # Initialize allocation matrix
    allocation = pd.DataFrame(0, index=customer.index, columns=facility.index)
    # Iterate over each customer
    for i in range(dist_mat.shape[0]):
        # Find the nearest facility with enough capacity among the ranked facilities
        for j in ranked_facilities[i]:
            if supply[j] >= demand[i]:
                # Allocate the customer to the chosen facility
                allocation.loc[customer.index[i], facility.index[j]] = 1 #demand[i]
                supply[j] -= demand[i]
                demand[i] = 0
                
                break
    return allocation


# Allocate customers to facilities using the constructive heuristic algorithm
allocation_SE = allocate_customers(dist_mat, demand, capacity, ranked_facilities)
cost_SE = calculate_cost(allocation_SE)

# do stuff
elapsed_SE = time.time() - t_SE

print(elapsed_SE)

##############################################################################
############### Greedy Adaptive Constructive Heuristics ######################
##############################################################################

# re define capacity and demand
capacity = facility['supply'].copy()
demand = customer['demand'].copy()

# Relaxation 
capacity = capacity

# Calculate distances between facilities and customers
dist_mat = distance.loc[customer.index, facility.index].values

# Initialize empty solution
allocation_GA = pd.DataFrame(np.zeros([len(customer.index), len(facility.index)]), index=customer.index, columns =facility.index)

# Initialize remaining customers, all customer node
remaining = customer.index


random.seed(111)

t_GA = time.time()

# While remaining customer is not zero, keep allocating
while len(remaining) > 0:
    # Create dummy variable to select random n customer from all customer
    dum_i = pd.DataFrame(np.zeros([len(remaining), 1]), index = remaining)
    
    # Sort customer by demand, largest demand will be on top
    for i in remaining:
        dum_i.loc[i] = customer.loc[i]['demand']
        
    dum_i = dum_i.sort_values(by=0, ascending=False)
    
    # Set parameter for selecting random number n for shortlist
    bi = 1
    bf = round(0.2*len(remaining)+1, 0)
    b = random.randint(bi,bf)
    
    # Shortlist the customer
    dum_i = dum_i[0:b]
    
    # Select a customer in the shortlist to be allocated
    i = dum_i.index[random.randint(0, len(dum_i)-1)]
    
    
    # Sorting facility by travel distance to the selected i
    dum_j = pd.DataFrame(np.zeros([len(facility.index), 1]), index=facility.index)
    for j in facility.index:
        dum_j.loc[j] = distance.loc[i][j]
    
    dum_j = dum_j.sort_values(by=0, ascending=True)
        
    # Allocate the customer
    for j in dum_j.index:
       if capacity[j] >= demand[i]:
          capacity[j] = capacity[j] - demand[i]
          demand[i] = 0
          break
            
        
    allocation_GA.loc[i,j] += 1
    
    # Remove selected customer from remaining customers
    remaining = remaining.drop(i)
    
cost_GA = calculate_cost(allocation_GA)  

elapsed_GA = time.time() - t_GA

print(elapsed_GA)



##############################################################################
################### Linear Optimization - Allocation #########################
##############################################################################

# Define problem
all_prob = pl.LpProblem("Facility_Allocation_Problem", pl.LpMinimize)

# Define variables
all_OM = pl.LpVariable.dicts("allocation", ((i,j) for i in customer.index for j in facility.index), lowBound=0, cat='Binary')

# Define objective function
all_prob += pl.lpSum([distance.loc[i][j] * all_OM[(i,j)] * customer.loc[i, 'demand'] for i in customer.index for j in facility.index])

# Define constraints

# Demand constraints - single sourcing. Customer can only be allocated to one facility
for i in customer.index:
    all_prob += pl.lpSum([all_OM[(i,j)] for j in facility.index]) == 1
    
# Supply constraints
for j in facility.index:
    all_prob += pl.lpSum([all_OM[(i,j)]*customer.loc[i, 'demand'] for i in customer.index]) <= facility.loc[j]['supply'] 


t_all_OM = time.time()

# Solve problem
all_prob.solve(pl.PULP_CBC_CMD(msg=1, maxSeconds=15))
#prob.solve(pl.CPLEX_PY())
#prob.solve(pl.GUROBI_CMD())

elapsed_all_OM = time.time() - t_all_OM
print(elapsed_all_OM)


# Print results
# print("Total Cost (OM) = ", pl.value(prob.objective))


allocation_OM = pd.DataFrame(np.zeros([len(customer.index), len(facility.index)]), index=customer.index, columns =facility.index)

for i in customer.index:
    for j in facility.index:
        if all_OM[(i,j)].varValue > 0:
            allocation_OM.loc[i,j] = 1

cost_all_OM = calculate_cost(allocation_OM)   



##############################################################################
########################### Capacity - Relaxation ############################
##############################################################################

##############################################################################
#################### Linear Optimization - Reallocation ######################
##############################################################################

# Relaxation 
capacity_AO = facility["supply"]*1.50

# Define reallocation cost as the paper suggested
reallocation_cost = 6



# Define problem
reall_prob = pl.LpProblem("Facility_Reallocation_Problem", pl.LpMinimize)

# Define variables
reall_OM = pl.LpVariable.dicts("allocation", [(i, j) for i in customer.index for j in facility.index], 0,1,
                             cat='Binary')

R = pl.LpVariable.dicts("reallocation", [i for i in customer.index], 0,1,
                             cat='Binary')

# Define objective function
reall_prob += pl.lpSum(
    [distance.loc[i][j] * reall_OM[i, j] * customer.loc[i, 'demand'] for i in customer.index for j in facility.index])

# Define constraints

# Demand constraints - single sourcing. Customer can only be allocated to one facility
for i in customer.index:
    reall_prob += pl.lpSum([reall_OM[i, j] for j in facility.index]) == 1

# Supply constraints
for j in facility.index:
    reall_prob += pl.lpSum(reall_OM[i, j] * customer.loc[i, 'demand'] for i in customer.index) <= capacity_AO[j]


for i in customer.index:
    reall_prob += R [i] == 1- pl.lpSum([allocation_GA.loc[i, j] * reall_OM[i, j] for j in facility.index])
    
    reall_prob += pl.lpSum(customer.loc[i,'demand'] * distance.loc[i,j]*
                           (allocation_GA.loc[i,j] - reall_OM[i, j]) for j in facility.index) >= R[i]*reallocation_cost



# Constraints that ensures that a customer is reallocated only when savings exceed costs
#for i in customer.index:
#    reall_prob += pl.lpSum([customer.loc[i, 'demand']*distance.loc[i, j]*(allocation_GA.loc[i, j]-reall_OM[(i, j)]) for j in
#                      facility.index]) >= (1-pl.lpSum(allocation_GA.loc[i, j]*reall_OM[(i, j)] for j in facility.index))* reallocation_cost



#for constraint in prob.constraints.values():
#    print(constraint)

t_reall_OM = time.time()

# Solve problem
reall_prob.solve(pl.PULP_CBC_CMD(msg=1, maxSeconds=15))
#prob.solve(pl.CPLEX_PY())
#prob.solve(pl.GUROBI_CMD())

elapsed_reall_OM = time.time() - t_reall_OM
print(elapsed_reall_OM)


# Print results
# print("Total Cost (OM) = ", pl.value(prob.objective))


reallocation_OM = pd.DataFrame(np.zeros([len(customer.index), len(facility.index)]), index=customer.index, columns =facility.index)

for i in customer.index:
    for j in facility.index:
        if reall_OM[(i,j)].varValue > 0:
            reallocation_OM.loc[i,j] = 1

cost_reall_OM = calculate_cost(reallocation_OM)   


##############################################################################

capacity_relax = capacity + facility["supply"]*0.50



##############################################################################
############### First Improvement Improvement Heuristics #####################
##############################################################################


# First Improvement Algorithm
def first_improvement(allocation):
    # Initialize variables
    improvement = True
    demand = customer['demand'].copy()
    #reallocation_cost = 0
    
    while improvement:
        improvement = False
        
        # Loop through each customer
        for i in customer.index:
            # Find current allocation of customer i
            current_facility = allocation.loc[i][allocation.loc[i]>0].index[0]
            
            # Find list of facilities sorted by distance to customer i
            ranked_facilities = distance.loc[i].sort_values()
            
            # Calculate current cost of customer i from GA
            current_cost = demand[i] * distance.loc[i, current_facility]
            
            # Loop through each facility in ranked list
            for j in ranked_facilities.index:
                # Skip current facility
                if j == current_facility:
                    continue
                
                # Calculate new cost of customer i to facility j
                new_cost = demand[i] * distance.loc[i, j]
                
                # Check if current cost - new cost is more than reallocation cost
                if current_cost - new_cost > reallocation_cost :
                    # Check if demand can be allocated to facility j
                    if capacity_relax[j] >= demand[i]:
                        # Allocate demand to facility j
                        allocation.loc[i][current_facility] = 0
                        allocation.loc[i][j] = 1
                    
                        # Update supply and demand
                        capacity_relax[current_facility] += demand[i]
                        capacity_relax[j] -= demand[i]
                    
                        #if new_cost < best_cost:
                        #best_allocation = allocation.copy()
                        improvement = True
                               
                        # Break out of loop once a facility has been found (first improvement)
                        break
                
                # If no facility can be found, move on to next customer
                if j == ranked_facilities.index[-1]:
                    break
                
    return allocation


            
t_FI = time.time()
allocation_FI = first_improvement(allocation_GA)
cost_FI = calculate_cost(allocation_FI)

elapsed_FI = time.time() - t_FI
print(elapsed_FI)








##############################################################################
########################## Print the Result ##################################
##############################################################################

# AO = Allocation with OM
print("Total cost (SE): " + '{:,.0f}'.format(round(cost_SE)))          
print("Total cost (GA): " + '{:,.0f}'.format(round(cost_GA)))    
print("Total cost (AO): " + '{:,.0f}'.format(round(cost_all_OM)))

# RO = Reallocation with OM 
print("Total cost (FI): " + '{:,.0f}'.format(round(cost_FI))) 
print("Total cost (RO): " + '{:,.0f}'.format(round(cost_reall_OM)))   

# AO = Allocation with OM
print(round(cost_SE))        
print(round(cost_GA))   
print(round(cost_all_OM))

print("")

# RO = Reallocation with OM 
print(round(cost_GA)) 
print(round(cost_FI))
print(round(cost_reall_OM))

    



print(round(elapsed_SE,2))          
print(round(elapsed_GA,2))  
print(round(elapsed_all_OM,2))
print(round(elapsed_FI,2))   
print(round(elapsed_reall_OM,2))


