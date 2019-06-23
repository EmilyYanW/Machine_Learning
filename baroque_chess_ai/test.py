#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:24:07 2019

@author: liuqingma
"""

import CardCaptor_Sakura_BC_Player_naiveZH as BC_player_naiveZH
import CardCaptor_Sakura_BC_Player_advancedZH as BC_player_advancedZH

import BC_state_etc as BC

currentState = BC.parse('''
c l i w k i l f 
p p p p p p p - 
- - - - - - - p 
- - P P - - - - 
P - I P P W P - 
- L - - - - - P 
- - - - - P - - 
F - - - K I L C
''')

currentState = BC.BC_state(currentState, BC.WHITE)

print("The current state is:")
print(currentState)

print("I'm working on testing and calculating, please wait a few minutes.......")

result = BC_player_naiveZH.parameterized_minimax(currentState, False, 2, False, False)
print("\nIf not using Alpha-beta pruning:")
print(result)

result = BC_player_naiveZH.parameterized_minimax(currentState, True, 2, False, False)
print("\nIf using Alpha-beta pruning:")
print(result)

result = BC_player_naiveZH.parameterized_minimax(currentState, True, 2, False, False)
print("\nIf not using Zobrist hashing:")
print(result)

result = BC_player_naiveZH.parameterized_minimax(currentState, True, 2, False, True)
print("\nIf using naive Zobrist hashing:")
print(result)

result = BC_player_advancedZH.parameterized_minimax(currentState, True, 2, False, True)
print("\nIf using advanced Zobrist hashing:")
print(result)

