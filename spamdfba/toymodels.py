from cobra import Model, Reaction, Metabolite
import cobra
import numpy as np
"""
A Toy Model is a Cobra Model with the following:

Toy_Model_SA

Reactions(NOT BALANCED):

-> S  Substrate uptake
S + ADP -> S_x + ATP  ATP production from catabolism
ATP -> ADP ATP maintenance
S_x + ATP -> X + ADP  Biomass production
S_x + ATP -> Amylase + ADP  Amylase production
Amylase -> Amylase Exchange
X -> Biomass Out
S_x + ADP -> P + ATP Metabolism stuff!
P ->  Product release

Metabolites:

P  Product
S  Substrate
S_x  Internal metabolite
X  Biomass
ADP  
ATP
Amylase

-----------------------------------------------------------------------


Toy_Model_NE_Aux_1:


EX_S_sp1: S -> lowerBound',-10,'upperBound',0
EX_A_sp1: A -> lowerBound',-100,'upperBound',100
EX_B_sp1: B -> lowerBound',-100,'upperBound',100
EX_P_sp1: P->  lowerBound',0,'upperBound',100
R_1_sp1: S  + 2 adp  -> P + 2 atp ,'lowerBound',0,'upperBound',Inf
R_2_sp1: P + atp  -> B  + adp 'lowerBound',0,'upperBound',Inf
R_3_sp1: P + 3 atp  -> A + 3 adp ,'lowerBound',0,'upperBound',Inf
R_4_sp1: 'atp -> adp  lowerBound',0,'upperBound',Inf
OBJ_sp1: 3 A + 3 B + 5 atp  -> 5 adp + biomass_sp1 lowerBound',0,'upperBound',Inf
Biomass_1 biomass_sp1  -> ','lowerBound',0,'upperBound',Inf,'objectiveCoef', 1);





Toy_Model_NE_Aux_2:


EX_S_sp1: S -> lowerBound',-10,'upperBound',0
EX_A_sp1: A -> lowerBound',-100,'upperBound',100
EX_B_sp1: B -> lowerBound',-100,'upperBound',100
EX_P_sp1: P->  lowerBound',0,'upperBound',100
R_1_sp1: S  + 2 adp  -> P + 2 atp ,'lowerBound',0,'upperBound',Inf
R_2_sp1: P + atp  -> B  + adp 'lowerBound',0,'upperBound',Inf
R_3_sp1: P + 3 atp  -> A + 3 adp ,'lowerBound',0,'upperBound',Inf
R_4_sp1: 'atp -> adp  lowerBound',0,'upperBound',Inf
OBJ_sp1: 3 A + 3 B + 5 atp  -> 5 adp + biomass_sp1 lowerBound',0,'upperBound',Inf
Biomass_1 biomass_sp1  -> ','lowerBound',0,'upperBound',Inf,'objectiveCoef', 1);

"""
ToyModel_SA = Model('Toy_Model')

### S_Uptake ###

S_Uptake = Reaction('Glc_e')
S = Metabolite('Glc', compartment='c')
S_Uptake.add_metabolites({S: -1})
S_Uptake.lower_bound = -20
S_Uptake.upper_bound = 0
ToyModel_SA.add_reactions([S_Uptake])

### ADP Production From Catabolism ###

ATP_Cat = Reaction('ATP_Cat')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
S_x = Metabolite('S_x', compartment='c')
ATP_Cat.add_metabolites({ADP: -1, S: -1, S_x: 1, ATP: 1})
ATP_Cat.lower_bound = 0
ATP_Cat.upper_bound = 1000
ToyModel_SA.add_reactions([ATP_Cat])

### ATP Maintenance ###

ATP_M = Reaction('ATP_M')
ATP_M.add_metabolites({ATP: -1, ADP: 1})
ATP_M.lower_bound = 1
ATP_M.upper_bound = 100
ToyModel_SA.add_reactions([ATP_M])

### Biomass Production ###

X = Metabolite('X', compartment='c')
X_Production = Reaction('X_Production')
X_Production.add_metabolites({S_x: -1, ATP: -100, ADP: 100, X: 1})
X_Production.lower_bound = 0
X_Production.upper_bound = 1000
ToyModel_SA.add_reactions([X_Production])

### Biomass Release ###

X_Release = Reaction('X_Ex')
X_Release.add_metabolites({X: -1})
X_Release.lower_bound = 0
X_Release.upper_bound = 1000
ToyModel_SA.add_reactions([X_Release])

### Metabolism stuff ###

P = Metabolite('P', compartment='c')
P_Prod = Reaction('P_Prod')
P_Prod.add_metabolites({S_x: -1, ATP: 1, ADP: -1, P: 0.1})
P_Prod.lower_bound = 0
P_Prod.upper_bound = 1000
ToyModel_SA.add_reactions([P_Prod])

### Product Release ###

P_out = Reaction('P_e')
P_out.add_metabolites({P: -1})
P_out.lower_bound = 0
P_out.upper_bound = 1000
ToyModel_SA.add_reactions([P_out])
ToyModel_SA.objective = 'X_Ex'

### Amylase Production ###
Amylase_Prod = Reaction('Amylase_Prod')
Amylase = Metabolite('Amylase', compartment='c')
Amylase_Prod.add_metabolites({S_x: -1, ATP: -1, ADP: 1, Amylase: 0.1})
Amylase_Prod.lower_bound = 0
Amylase_Prod.upper_bound = 1000
ToyModel_SA.add_reactions([Amylase_Prod])

### Amylase Exchange ###
Amylase_Ex = Reaction('Amylase_e')
Amylase_Ex.add_metabolites({Amylase: -1})
Amylase_Ex.lower_bound = 0
Amylase_Ex.upper_bound = 1000
ToyModel_SA.add_reactions([Amylase_Ex])

ToyModel_SA.biomass_ind=4
ToyModel_SA.exchange_reactions=tuple([ToyModel_SA.reactions.index(i) for i in ToyModel_SA.exchanges])



#########################################################
#########################################################

### S_Uptake ###
Toy_Model_NE_Aux_1 = Model('Toy_1_Aux')
					   
EX_S_sp1 = Reaction('S_e')
S = Metabolite('S', compartment='c')
EX_S_sp1.add_metabolites({S: -1})
EX_S_sp1.lower_bound = -10
EX_S_sp1.upper_bound = 0
Toy_Model_NE_Aux_1.add_reactions([EX_S_sp1])


EX_A_sp1 = Reaction('A_e')
A = Metabolite('A', compartment='c')
EX_A_sp1.add_metabolites({A: -1})
EX_A_sp1.lower_bound = -100
EX_A_sp1.upper_bound = 100
Toy_Model_NE_Aux_1.add_reactions([EX_A_sp1])


EX_B_sp1 = Reaction('B_e')
B = Metabolite('B', compartment='c')
EX_B_sp1.add_metabolites({B: -1})
EX_B_sp1.lower_bound = 0
EX_B_sp1.upper_bound = 100
Toy_Model_NE_Aux_1.add_reactions([EX_B_sp1])



EX_P_sp1 = Reaction('P_e')
P = Metabolite('P', compartment='c')
EX_P_sp1.add_metabolites({P:-1})
EX_P_sp1.lower_bound = 0
EX_P_sp1.upper_bound = 100
Toy_Model_NE_Aux_1.add_reactions([EX_P_sp1])


R_1_sp1 = Reaction('R_1_sp1')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
R_1_sp1.add_metabolites({ADP: -2, S: -1, P: 1, ATP: 2})
R_1_sp1.lower_bound = 0
R_1_sp1.upper_bound = 1000
Toy_Model_NE_Aux_1.add_reactions([R_1_sp1])


R_2_sp1 = Reaction('R_2_sp1')
R_2_sp1.add_metabolites({ADP: 1, P: -1, B: 3, ATP: -1})
R_2_sp1.lower_bound = 0
R_2_sp1.upper_bound = 1000
Toy_Model_NE_Aux_1.add_reactions([R_2_sp1])


# R_3_sp1 = Reaction('R_3_sp1')
# R_3_sp1.add_metabolites({ADP: 3, P: -1, A: 1, ATP: -3})
# R_3_sp1.lower_bound = 0
# R_3_sp1.upper_bound = 1000
# Toy_Model_NE_Aux_1.add_reactions([R_3_sp1])



R_4_sp1 = Reaction('R_4_sp1')
R_4_sp1.add_metabolites({ADP:1 ,ATP: -1})
R_4_sp1.lower_bound = 0
R_4_sp1.upper_bound = 1000
Toy_Model_NE_Aux_1.add_reactions([R_4_sp1])


OBJ_sp1 = Reaction("OBJ_sp1")
biomass_sp1 = Metabolite('biomass_sp1', compartment='c')
OBJ_sp1.add_metabolites({ADP:5 ,ATP: -5,biomass_sp1:0.1,A:-5,B:-5})
OBJ_sp1.lower_bound = 0
OBJ_sp1.upper_bound = 1000
Toy_Model_NE_Aux_1.add_reactions([OBJ_sp1])

Biomass_1 = Reaction("Biomass_1")
Biomass_1.add_metabolites({biomass_sp1:-1})
Biomass_1.lower_bound = 0
Biomass_1.upper_bound = 1000
Toy_Model_NE_Aux_1.add_reactions([Biomass_1])

Toy_Model_NE_Aux_1.objective='Biomass_1'
Toy_Model_NE_Aux_1.biomass_ind=8
Toy_Model_NE_Aux_1.exchange_reactions=tuple([Toy_Model_NE_Aux_1.reactions.index(i) for i in Toy_Model_NE_Aux_1.exchanges])


### ADP Production From Catabolism ###

Toy_Model_NE_Aux_2 = Model('Toy_2_Aux')

### S_Uptake ###

EX_S_sp2 = Reaction('S_e')
S = Metabolite('S', compartment='c')
EX_S_sp2.add_metabolites({S: -1})
EX_S_sp2.lower_bound = -10
EX_S_sp2.upper_bound = 0
Toy_Model_NE_Aux_2.add_reactions([EX_S_sp2])


EX_A_sp2 = Reaction('A_e')
A = Metabolite('A', compartment='c')
EX_A_sp2.add_metabolites({A: -1})
EX_A_sp2.lower_bound = 0
EX_A_sp2.upper_bound = 100
Toy_Model_NE_Aux_2.add_reactions([EX_A_sp2])


EX_B_sp2 = Reaction('B_e')
B = Metabolite('B', compartment='c')
EX_B_sp2.add_metabolites({B: -1})
EX_B_sp2.lower_bound = -100
EX_B_sp2.upper_bound = 100
Toy_Model_NE_Aux_2.add_reactions([EX_B_sp2])



EX_P_sp2 = Reaction('P_e')
P = Metabolite('P', compartment='c')
EX_P_sp2.add_metabolites({P:-1})
EX_P_sp2.lower_bound = 0
EX_P_sp2.upper_bound = 100
Toy_Model_NE_Aux_2.add_reactions([EX_P_sp2])


R_1_sp2 = Reaction('R_1_sp2')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
R_1_sp2.add_metabolites({ADP: -2, S: -1, P: 1, ATP: 2})
R_1_sp2.lower_bound = 0
R_1_sp2.upper_bound = 1000
Toy_Model_NE_Aux_2.add_reactions([R_1_sp2])


# R_2_sp2 = Reaction('R_2_sp2')
# R_2_sp2.add_metabolites({ADP: 3, P: -1, B: 1, ATP: -3})
# R_2_sp2.lower_bound = 0
# R_2_sp2.upper_bound = 1000
# Toy_Model_NE_Aux_2.add_reactions([R_2_sp2])


R_3_sp2 = Reaction('R_3_sp2')
R_3_sp2.add_metabolites({ADP: 1, P: -1, A: 3, ATP: -1})
R_3_sp2.lower_bound = 0
R_3_sp2.upper_bound = 1000
Toy_Model_NE_Aux_2.add_reactions([R_3_sp2])



R_4_sp2 = Reaction('R_4_sp2')
R_4_sp2.add_metabolites({ADP:1 ,ATP: -1})
R_4_sp2.lower_bound = 0
R_4_sp2.upper_bound = 1000
Toy_Model_NE_Aux_2.add_reactions([R_4_sp2])




OBJ_sp2 = Reaction("OBJ_sp2")
biomass_sp2 = Metabolite('biomass_sp2', compartment='c')
OBJ_sp2.add_metabolites({ADP:5 ,ATP: -5,biomass_sp2:0.1,A:-5,B:-5})
OBJ_sp2.lower_bound = 0
OBJ_sp2.upper_bound = 1000
Toy_Model_NE_Aux_2.add_reactions([OBJ_sp2])

Biomass_2 = Reaction("Biomass_2")
Biomass_2.add_metabolites({biomass_sp2:-1})
Biomass_2.lower_bound = 0
Biomass_2.upper_bound = 1000
Toy_Model_NE_Aux_2.add_reactions([Biomass_2])
Toy_Model_NE_Aux_2.objective="Biomass_2"
Toy_Model_NE_Aux_2.biomass_ind=8
Toy_Model_NE_Aux_2.exchange_reactions=tuple([Toy_Model_NE_Aux_2.reactions.index(i) for i in Toy_Model_NE_Aux_2.exchanges])



########################## Mutualistic Species ##########################

### S_Uptake ###
Toy_Model_NE_Mut_1 = Model('Toy_1_Mut')
					   
EX_S_sp1 = Reaction('S_e')
S = Metabolite('S', compartment='c')
EX_S_sp1.add_metabolites({S: -1})
EX_S_sp1.lower_bound = -10
EX_S_sp1.upper_bound = 0
Toy_Model_NE_Mut_1.add_reactions([EX_S_sp1])


EX_A_sp1 = Reaction('A_e')
A = Metabolite('A', compartment='c')
EX_A_sp1.add_metabolites({A: -1})
EX_A_sp1.lower_bound = -100
EX_A_sp1.upper_bound = 100
Toy_Model_NE_Mut_1.add_reactions([EX_A_sp1])


EX_B_sp1 = Reaction('B_e')
B = Metabolite('B', compartment='c')
EX_B_sp1.add_metabolites({B: -1})
EX_B_sp1.lower_bound = -100
EX_B_sp1.upper_bound = 100
Toy_Model_NE_Mut_1.add_reactions([EX_B_sp1])



EX_P_sp1 = Reaction('P_e')
P = Metabolite('P', compartment='c')
EX_P_sp1.add_metabolites({P:-1})
EX_P_sp1.lower_bound = 0
EX_P_sp1.upper_bound = 100
Toy_Model_NE_Mut_1.add_reactions([EX_P_sp1])


R_1_sp1 = Reaction('R_1_sp1')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
R_1_sp1.add_metabolites({ADP: -2, S: -1, P: 1, ATP: 2})
R_1_sp1.lower_bound = 0
R_1_sp1.upper_bound = 1000
Toy_Model_NE_Mut_1.add_reactions([R_1_sp1])


R_2_sp1 = Reaction('R_2_sp1')
R_2_sp1.add_metabolites({ADP: 1, P: -1, B: 3, ATP: -1})
R_2_sp1.lower_bound = 0
R_2_sp1.upper_bound = 1000
Toy_Model_NE_Mut_1.add_reactions([(R_2_sp1)])


R_3_sp1 = Reaction('R_3_sp1')
R_3_sp1.add_metabolites({ADP: 3, P: -1, A: 1, ATP: -3})
R_3_sp1.lower_bound = 0
R_3_sp1.upper_bound = 1000
Toy_Model_NE_Mut_1.add_reactions([R_3_sp1])



R_4_sp1 = Reaction('R_4_sp1')
R_4_sp1.add_metabolites({ADP:1 ,ATP: -1})
R_4_sp1.lower_bound = 0
R_4_sp1.upper_bound = 1000
Toy_Model_NE_Mut_1.add_reactions([R_4_sp1])


OBJ_sp1 = Reaction("OBJ_sp1")
biomass_sp1 = Metabolite('biomass_sp1', compartment='c')
OBJ_sp1.add_metabolites({ADP:5 ,ATP: -5,biomass_sp1:0.1,A:-5,B:-5})
OBJ_sp1.lower_bound = 0
OBJ_sp1.upper_bound = 1000
Toy_Model_NE_Mut_1.add_reactions([OBJ_sp1])

Biomass_1 = Reaction("Biomass_1")
Biomass_1.add_metabolites({biomass_sp1:-1})
Biomass_1.lower_bound = 0
Biomass_1.upper_bound = 1000
Toy_Model_NE_Mut_1.add_reactions([Biomass_1])

Toy_Model_NE_Mut_1.objective='Biomass_1'
Toy_Model_NE_Mut_1.biomass_ind=9
Toy_Model_NE_Mut_1.exchange_reactions=tuple([Toy_Model_NE_Mut_1.reactions.index(i) for i in Toy_Model_NE_Mut_1.exchanges])



Toy_Model_NE_Mut_2 = Model('Toy_2_Mut')

EX_S_sp2 = Reaction('S_e')
S = Metabolite('S', compartment='c')
EX_S_sp2.add_metabolites({S: -1})
EX_S_sp2.lower_bound = -10
EX_S_sp2.upper_bound = 0
Toy_Model_NE_Mut_2.add_reactions([EX_S_sp2])


EX_A_sp2 = Reaction('A_e')
A = Metabolite('A', compartment='c')
EX_A_sp2.add_metabolites({A: -1})
EX_A_sp2.lower_bound = -100
EX_A_sp2.upper_bound = 100
Toy_Model_NE_Mut_2.add_reactions([EX_A_sp2])


EX_B_sp2 = Reaction('B_e')
B = Metabolite('B', compartment='c')
EX_B_sp2.add_metabolites({B: -1})
EX_B_sp2.lower_bound = -100
EX_B_sp2.upper_bound = 100
Toy_Model_NE_Mut_2.add_reactions([EX_B_sp2])


EX_P_sp2 = Reaction('P_e')
P = Metabolite('P', compartment='c')
EX_P_sp2.add_metabolites({P:-1})
EX_P_sp2.lower_bound = 0
EX_P_sp2.upper_bound = 100
Toy_Model_NE_Mut_2.add_reactions([EX_P_sp2])


R_1_sp2 = Reaction('R_1_sp2')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
R_1_sp2.add_metabolites({ADP: -2, S: -1, P: 1, ATP: 2})
R_1_sp2.lower_bound = 0
R_1_sp2.upper_bound = 1000
Toy_Model_NE_Mut_2.add_reactions([R_1_sp2])


R_2_sp2 = Reaction('R_2_sp2')
R_2_sp2.add_metabolites({ADP: 1, P: -1, B: 3, ATP: -1})
R_2_sp2.lower_bound = 0
R_2_sp2.upper_bound = 1000
Toy_Model_NE_Mut_2.add_reactions([(R_2_sp2)])

R_3_sp2 = Reaction('R_3_sp2')
R_3_sp2.add_metabolites({ADP: 3, P: -1, A: 1, ATP: -3})
R_3_sp2.lower_bound = 0
R_3_sp2.upper_bound = 1000
Toy_Model_NE_Mut_2.add_reactions([R_3_sp2])

R_4_sp2 = Reaction('R_4_sp2')
R_4_sp2.add_metabolites({ADP:1 ,ATP: -1})
R_4_sp2.lower_bound = 0
R_4_sp2.upper_bound = 1000
Toy_Model_NE_Mut_2.add_reactions([R_4_sp2])

OBJ_sp2 = Reaction("OBJ_sp2")
biomass_sp2 = Metabolite('biomass_sp2', compartment='c')
OBJ_sp2.add_metabolites({ADP:5 ,ATP: -5,biomass_sp2:0.1,A:-5,B:-5})
OBJ_sp2.lower_bound = 0
OBJ_sp2.upper_bound = 1000
Toy_Model_NE_Mut_2.add_reactions([OBJ_sp2])

Biomass_2 = Reaction("Biomass_2")
Biomass_2.add_metabolites({biomass_sp2:-1})
Biomass_2.lower_bound = 0
Biomass_2.upper_bound = 1000
Toy_Model_NE_Mut_2.add_reactions([Biomass_2])

Toy_Model_NE_Mut_2.objective='Biomass_2'
Toy_Model_NE_Mut_2.biomass_ind=9
Toy_Model_NE_Mut_2.exchange_reactions=tuple([Toy_Model_NE_Mut_2.reactions.index(i) for i in Toy_Model_NE_Mut_2.exchanges])



ToyModel_SA_1 = Model('Toy_Model1')

### S_Uptake ###

S_Uptake = Reaction('Glc_1_e')
S = Metabolite('Glc_1', compartment='c')
S_Uptake.add_metabolites({S: -1})
S_Uptake.lower_bound = -20
S_Uptake.upper_bound = 0
ToyModel_SA_1.add_reactions([S_Uptake])

### ADP Production From Catabolism ###

ATP_Cat = Reaction('ATP_Cat')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
S_x = Metabolite('S_x', compartment='c')
ATP_Cat.add_metabolites({ADP: -1, S: -1, S_x: 1, ATP: 1})
ATP_Cat.lower_bound = 0
ATP_Cat.upper_bound = 1000
ToyModel_SA_1.add_reactions([ATP_Cat])

### ATP Maintenance ###

ATP_M = Reaction('ATP_M')
ATP_M.add_metabolites({ATP: -1, ADP: 1})
ATP_M.lower_bound = 1
ATP_M.upper_bound = 100
ToyModel_SA_1.add_reactions([ATP_M])

### Biomass Production ###

X = Metabolite('X', compartment='c')
X_Production = Reaction('X_Production')
X_Production.add_metabolites({S_x: -1, ATP: -100, ADP: 100, X: 1})
X_Production.lower_bound = 0
X_Production.upper_bound = 1000
ToyModel_SA_1.add_reactions([X_Production])

### Biomass Release ###

X_Release = Reaction('X_Ex')
X_Release.add_metabolites({X: -1})
X_Release.lower_bound = 0
X_Release.upper_bound = 1000
ToyModel_SA_1.add_reactions([X_Release])

### Metabolism stuff ###

P = Metabolite('P', compartment='c')
P_Prod = Reaction('P_Prod')
P_Prod.add_metabolites({S_x: -1, ATP: 1, ADP: -1, P: 0.1})
P_Prod.lower_bound = 0
P_Prod.upper_bound = 1000
ToyModel_SA_1.add_reactions([P_Prod])

### Product Release ###

P_out = Reaction('P_e')
P_out.add_metabolites({P: -1})
P_out.lower_bound = 0
P_out.upper_bound = 1000
ToyModel_SA_1.add_reactions([P_out])
ToyModel_SA_1.objective = 'X_Ex'

### Amylase Production ###
Amylase_Prod = Reaction('Amylase_Prod')
Amylase_1 = Metabolite('Amylase_1', compartment='c')
Amylase_Prod.add_metabolites({S_x: -1, ATP: -1, ADP: 1, Amylase_1: 0.1})
Amylase_Prod.lower_bound = 0
Amylase_Prod.upper_bound = 1000
ToyModel_SA_1.add_reactions([Amylase_Prod])

### Amylase Exchange ###
Amylase_Ex = Reaction('Amylase_1_e')
Amylase_Ex.add_metabolites({Amylase_1: -1})
Amylase_Ex.lower_bound = 0
Amylase_Ex.upper_bound = 1000
ToyModel_SA_1.add_reactions([Amylase_Ex])

ToyModel_SA_1.biomass_ind=4
ToyModel_SA_1.exchange_reactions=tuple([ToyModel_SA_1.reactions.index(i) for i in ToyModel_SA_1.exchanges])

ToyModel_SA_2 = Model('Toy_Model2')
S_Uptake = Reaction('Glc_2_e')
S = Metabolite('Glc_2', compartment='c')
S_Uptake.add_metabolites({S: -1})
S_Uptake.lower_bound = -20
S_Uptake.upper_bound = 0
ToyModel_SA_2.add_reactions([S_Uptake])

### ADP Production From Catabolism ###
ATP_Cat = Reaction('ATP_Cat')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
S_x = Metabolite('S_x', compartment='c')
ATP_Cat.add_metabolites({ADP: -1, S: -1, S_x: 1, ATP: 1})
ATP_Cat.lower_bound = 0
ATP_Cat.upper_bound = 1000
ToyModel_SA_2.add_reactions([ATP_Cat])

### ATP Maintenance ###
ATP_M = Reaction('ATP_M')
ATP_M.add_metabolites({ATP: -1, ADP: 1})
ATP_M.lower_bound = 1
ATP_M.upper_bound = 100
ToyModel_SA_2.add_reactions([ATP_M])

### Biomass Production ###
X = Metabolite('X', compartment='c')
X_Production = Reaction('X_Production')
X_Production.add_metabolites({S_x: -1, ATP: -100, ADP: 100, X: 1})
X_Production.lower_bound = 0
X_Production.upper_bound = 1000
ToyModel_SA_2.add_reactions([X_Production])

### Biomass Release ###
X_Release = Reaction('X_Ex')
X_Release.add_metabolites({X: -1})
X_Release.lower_bound = 0
X_Release.upper_bound = 1000
ToyModel_SA_2.add_reactions([X_Release])

### Metabolism stuff ###
P = Metabolite('P', compartment='c')
P_Prod = Reaction('P_Prod')
P_Prod.add_metabolites({S_x: -1, ATP: 1, ADP: -1, P: 0.1})
P_Prod.lower_bound = 0
P_Prod.upper_bound = 1000
ToyModel_SA_2.add_reactions([P_Prod])

### Product Release ###
P_out = Reaction('P_e')
P_out.add_metabolites({P: -1})
P_out.lower_bound = 0
P_out.upper_bound = 1000
ToyModel_SA_2.add_reactions([P_out])
ToyModel_SA_2.objective = 'X_Ex'

### Amylase Production ###
Amylase_Prod = Reaction('Amylase_Prod')
Amylase_2 = Metabolite('Amylase_2', compartment='c')
Amylase_Prod.add_metabolites({S_x: -1, ATP: -1, ADP: 1, Amylase_2: 0.1})
Amylase_Prod.lower_bound = 0
Amylase_Prod.upper_bound = 1000
ToyModel_SA_2.add_reactions([Amylase_Prod])

### Amylase Exchange ###
Amylase_Ex = Reaction('Amylase_2_e')
Amylase_Ex.add_metabolites({Amylase_2: -1})
Amylase_Ex.lower_bound = 0
Amylase_Ex.upper_bound = 1000
ToyModel_SA_2.add_reactions([Amylase_Ex])

ToyModel_SA_2.biomass_ind=4
ToyModel_SA_2.exchange_reactions=tuple([ToyModel_SA_2.reactions.index(i) for i in ToyModel_SA_2.exchanges])

ToyModel_SA_3 = Model('Toy_Model3')
S_Uptake = Reaction('Glc_3_e')
S = Metabolite('Glc_3', compartment='c')
S_Uptake.add_metabolites({S: -1})
S_Uptake.lower_bound = -20
S_Uptake.upper_bound = 0
ToyModel_SA_3.add_reactions([S_Uptake])

### ADP Production From Catabolism ###
ATP_Cat = Reaction('ATP_Cat')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
S_x = Metabolite('S_x', compartment='c')
ATP_Cat.add_metabolites({ADP: -1, S: -1, S_x: 1, ATP: 1})
ATP_Cat.lower_bound = 0
ATP_Cat.upper_bound = 1000

ToyModel_SA_3.add_reactions([ATP_Cat])

### ATP Maintenance ###
ATP_M = Reaction('ATP_M')
ATP_M.add_metabolites({ATP: -1, ADP: 1})
ATP_M.lower_bound = 1
ATP_M.upper_bound = 100
ToyModel_SA_3.add_reactions([ATP_M])

### Biomass Production ###
X = Metabolite('X', compartment='c')
X_Production = Reaction('X_Production')
X_Production.add_metabolites({S_x: -1, ATP: -100, ADP: 100, X: 1})  
X_Production.lower_bound = 0
X_Production.upper_bound = 1000
ToyModel_SA_3.add_reactions([X_Production])

### Biomass Release ###
X_Release = Reaction('X_Ex')
X_Release.add_metabolites({X: -1})
X_Release.lower_bound = 0
X_Release.upper_bound = 1000
ToyModel_SA_3.add_reactions([X_Release])

### Metabolism stuff ###
P = Metabolite('P', compartment='c')
P_Prod = Reaction('P_Prod')
P_Prod.add_metabolites({S_x: -1, ATP: 1, ADP: -1, P: 0.1})
P_Prod.lower_bound = 0
P_Prod.upper_bound = 1000
ToyModel_SA_3.add_reactions([P_Prod])

### Product Release ###
P_out = Reaction('P_e')
P_out.add_metabolites({P: -1})
P_out.lower_bound = 0
P_out.upper_bound = 1000
ToyModel_SA_3.add_reactions([P_out])
ToyModel_SA_3.objective = 'X_Ex'

### Amylase Production ###
Amylase_Prod = Reaction('Amylase_Prod')
Amylase_3 = Metabolite('Amylase_3', compartment='c')
Amylase_Prod.add_metabolites({S_x: -1, ATP: -1, ADP: 1, Amylase_3: 0.1})
Amylase_Prod.lower_bound = 0
Amylase_Prod.upper_bound = 1000
ToyModel_SA_3.add_reactions([Amylase_Prod])

### Amylase Exchange ###
Amylase_Ex = Reaction('Amylase_3_e')
Amylase_Ex.add_metabolites({Amylase_3: -1})
Amylase_Ex.lower_bound = 0
Amylase_Ex.upper_bound = 1000
ToyModel_SA_3.add_reactions([Amylase_Ex])

ToyModel_SA_3.biomass_ind=4
ToyModel_SA_3.exchange_reactions=tuple([ToyModel_SA_3.reactions.index(i) for i in ToyModel_SA_3.exchanges])

ToyModel_SA_4 = Model('Toy_Model4')
S_Uptake = Reaction('Glc_4_e')
S = Metabolite('Glc_4', compartment='c')  
S_Uptake.add_metabolites({S: -1})
S_Uptake.lower_bound = -20
S_Uptake.upper_bound = 0
ToyModel_SA_4.add_reactions([S_Uptake])

### ADP Production From Catabolism ###
ATP_Cat = Reaction('ATP_Cat')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
S_x = Metabolite('S_x', compartment='c')
ATP_Cat.add_metabolites({ADP: -1, S: -1, S_x: 1, ATP: 1})
ATP_Cat.lower_bound = 0
ATP_Cat.upper_bound = 1000

ToyModel_SA_4.add_reactions([ATP_Cat])

### ATP Maintenance ###
ATP_M = Reaction('ATP_M')
ATP_M.add_metabolites({ATP: -1, ADP: 1})
ATP_M.lower_bound = 1
ATP_M.upper_bound = 100
ToyModel_SA_4.add_reactions([ATP_M])

### Biomass Production ###
X = Metabolite('X', compartment='c')
X_Production = Reaction('X_Production')
X_Production.add_metabolites({S_x: -1, ATP: -100, ADP: 100, X: 1})
X_Production.lower_bound = 0
X_Production.upper_bound = 1000
ToyModel_SA_4.add_reactions([X_Production])

### Biomass Release ###
X_Release = Reaction('X_Ex')
X_Release.add_metabolites({X: -1})
X_Release.lower_bound = 0
X_Release.upper_bound = 1000
ToyModel_SA_4.add_reactions([X_Release])

### Metabolism stuff ###
P = Metabolite('P', compartment='c')
P_Prod = Reaction('P_Prod')
P_Prod.add_metabolites({S_x: -1, ATP: 1, ADP: -1, P: 0.1})
P_Prod.lower_bound = 0
P_Prod.upper_bound = 1000
ToyModel_SA_4.add_reactions([P_Prod])

### Product Release ###
P_out = Reaction('P_e')
P_out.add_metabolites({P: -1})
P_out.lower_bound = 0
P_out.upper_bound = 1000
ToyModel_SA_4.add_reactions([P_out])
ToyModel_SA_4.objective = 'X_Ex'

### Amylase Production ###
Amylase_Prod = Reaction('Amylase_Prod')
Amylase_4 = Metabolite('Amylase_4', compartment='c')

Amylase_Prod.add_metabolites({S_x: -1, ATP: -1, ADP: 1, Amylase_4: 0.1})
Amylase_Prod.lower_bound = 0
Amylase_Prod.upper_bound = 1000
ToyModel_SA_4.add_reactions([Amylase_Prod])

### Amylase Exchange ###
Amylase_Ex = Reaction('Amylase_4_e')
Amylase_Ex.add_metabolites({Amylase_4: -1})
Amylase_Ex.lower_bound = 0
Amylase_Ex.upper_bound = 1000
ToyModel_SA_4.add_reactions([Amylase_Ex])

ToyModel_SA_4.biomass_ind=4
ToyModel_SA_4.exchange_reactions=tuple([ToyModel_SA_4.reactions.index(i) for i in ToyModel_SA_4.exchanges])

ToyModel_SA_5 = Model('Toy_Model5')
S_Uptake = Reaction('Glc_5_e')
S = Metabolite('Glc_5', compartment='c')
S_Uptake.add_metabolites({S: -1})
S_Uptake.lower_bound = -20
S_Uptake.upper_bound = 0
ToyModel_SA_5.add_reactions([S_Uptake])

### ADP Production From Catabolism ###
ATP_Cat = Reaction('ATP_Cat')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
S_x = Metabolite('S_x', compartment='c')
ATP_Cat.add_metabolites({ADP: -1, S: -1, S_x: 1, ATP: 1})
ATP_Cat.lower_bound = 0
ATP_Cat.upper_bound = 1000

ToyModel_SA_5.add_reactions([ATP_Cat])

### ATP Maintenance ###
ATP_M = Reaction('ATP_M')

ATP_M.add_metabolites({ATP: -1, ADP: 1})
ATP_M.lower_bound = 1
ATP_M.upper_bound = 100
ToyModel_SA_5.add_reactions([ATP_M])

### Biomass Production ###
X = Metabolite('X', compartment='c')
X_Production = Reaction('X_Production')

X_Production.add_metabolites({S_x: -1, ATP: -100, ADP: 100, X: 1})
X_Production.lower_bound = 0
X_Production.upper_bound = 1000
ToyModel_SA_5.add_reactions([X_Production])

### Biomass Release ###
X_Release = Reaction('X_Ex')
X_Release.add_metabolites({X: -1})
X_Release.lower_bound = 0
X_Release.upper_bound = 1000
ToyModel_SA_5.add_reactions([X_Release])

### Metabolism stuff ###
P = Metabolite('P', compartment='c')
P_Prod = Reaction('P_Prod')

P_Prod.add_metabolites({S_x: -1, ATP: 1, ADP: -1, P: 0.1})
P_Prod.lower_bound = 0
P_Prod.upper_bound = 1000
ToyModel_SA_5.add_reactions([P_Prod])

### Product Release ###
P_out = Reaction('P_e')
P_out.add_metabolites({P: -1})
P_out.lower_bound = 0
P_out.upper_bound = 1000
ToyModel_SA_5.add_reactions([P_out])
ToyModel_SA_5.objective = 'X_Ex'

### Amylase Production ###
Amylase_Prod = Reaction('Amylase_Prod')
Amylase_5 = Metabolite('Amylase_5', compartment='c')

Amylase_Prod.add_metabolites({S_x: -1, ATP: -1, ADP: 1, Amylase_5: 0.1})
Amylase_Prod.lower_bound = 0
Amylase_Prod.upper_bound = 1000
ToyModel_SA_5.add_reactions([Amylase_Prod])

### Amylase Exchange ###
Amylase_Ex = Reaction('Amylase_5_e')
Amylase_Ex.add_metabolites({Amylase_5: -1})
Amylase_Ex.lower_bound = 0
Amylase_Ex.upper_bound = 1000
ToyModel_SA_5.add_reactions([Amylase_Ex])

ToyModel_SA_5.biomass_ind=5
ToyModel_SA_5.exchange_reactions=tuple([ToyModel_SA_5.reactions.index(i) for i in ToyModel_SA_5.exchanges])




if __name__ == '__main__':
	# print(ToyModel_SA.optimize().fluxes)
	# print(ToyModel_SA.optimize().status)
	# print(ToyModel_SA.exchanges)
	
	print(Toy_Model_NE_Aux_1.optimize().fluxes)
	print(Toy_Model_NE_Aux_1.optimize().status)
	print(Toy_Model_NE_Aux_1.exchanges)

	print(Toy_Model_NE_Aux_2.optimize().fluxes)
	print(Toy_Model_NE_Aux_2.optimize().status)
	print(Toy_Model_NE_Aux_2.exchanges)

	# print(Toy_Model_NE_Mut_1.optimize().fluxes)
	# print(Toy_Model_NE_Mut_1.optimize().status)
	# print(Toy_Model_NE_Mut_1.exchanges)

	# print(Toy_Model_NE_Mut_2.optimize().fluxes)
	# print(Toy_Model_NE_Mut_2.optimize().status)
	# print(Toy_Model_NE_Mut_2.exchanges)	
