'''BC_checker.py
This module provides two methods for refereeing
Baroque Chess games, using a web service hosted
at the Univ. of Washington's Paul G. Allen School
of Computer Science and Engineering.

The service is hosted on a web server called xanthippe.
The two methods are:
(a) validate_move (for verifying that a given move is legal)
(b) any_moves (for determining if there are ANY legal 
  moves from a given state).  If not, this typically means
  the game is a draw.

S. Tanimoto, May 5, 2019. (Based on prev. devel. of
to_bc_move_validator.py, April 8, 2019)

'''

import urllib.request   
import urllib.parse      
URL = "http://xanthippe.cs.washington.edu/bc/bcpyserv/bc-validate.php"

def validate_move(starting_square, board1_string, board2_string):
  params = {'service':'validate',\
            'move':starting_square,\
            'state1':board1_string,\
            'state2':board2_string}
  return handle_query(params)

def any_moves(board_string, whose_move=0):
  params = {'service':'anymoves',\
            'whose_move':whose_move,\
            'state1':board_string}
  return handle_query(params)

def handle_query(params):
  query_string = urllib.parse.urlencode( params )   
  data = query_string.encode( "ascii" )      
  with urllib.request.urlopen( URL, data ) as response:        
    response_text = response.read()
    response_text = response_text.decode('utf-8') 
    #print("response_text is: "+response_text)
    status, comment = eval(response_text)
    return((status, comment))

def remove_last_2_lines_from_string(s):
    s = s[:s.rfind('\n')]
    s = s[:s.rfind('\n')]
    return s+'\n'

def board_only(state):
  '''Helper function for BaroqueGameMaster.py.
  Returns the string representation of the board,
  without the whose_move part.'''
#  import BC_state_etc as bc
#  b = bc.__repr__
  b = state.__repr__()
  #print("In BC_check.board_only: ")
  b = remove_last_2_lines_from_string(b)
  #print(b)
  return b
