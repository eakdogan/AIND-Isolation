"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    #print('own,opp moves'+str(len(own_moves))+'-'+str(len(opp_moves)))
    #print ('score returned: ' +str(float(len(own_moves) - 2 * len(opp_moves)) ))

    """Below are some of the heuristics tested"""
    return float(2*len(own_moves) - 3 * len(opp_moves))  #intuition about the L move
    #return random.uniform(0, 1)
    #return float(3*len(own_moves) - 2 * len(opp_moves))
    #return float(len(own_moves) - 2 * len(opp_moves))#
    #return (100-len(game.get_blank_spaces())*(float(len(own_moves)) * len(opp_moves)))
    #return (float(len(own_moves) - (1/(len(game.get_blank_spaces())+1)) * len(opp_moves)))
    #return float(len(own_moves) - 3 * len(opp_moves))


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        """Perform required initializations, select an initial
        move from the game board, or return immediately if there are no legal moves"""
        if (not legal_moves ): return (-1, -1)
        """Prefer sort of middle for start position"""
        if (game.move_count == 0): return (4, 4)
        """Initialize a value and move and try to find better in the deepening search"""
        next_move = legal_moves[0]
        current_value = float('-inf') #try to find better and overwrite this value

        """Check time left first, and return a move immediately if critic"""
        if (self.time_left() < self.TIMER_THRESHOLD + 3):
            #raise Timeout()
            #print('time left: '+str(self.time_left()))
            return next_move

        """Run minimax or alphabeta and finish for non-iterative"""
        if (self.iterative == False):
            """"Perform the search for value"""
            if (self.method =="minimax"):
                result = self.minimax(game, self.search_depth, True)
                value = result[0]
                new_move = result[1]
            if (self.method =="alphabeta"):
                result = self.alphabeta(game, self.search_depth,float("-inf"),float("inf"), True)
                value = result[0]
                new_move = result[1]
            return new_move
        """Call (alpha beta or minimax) repeatedly deepr. Handle overtime in exception"""
        """Keep looping for deepening and searching better values for iterative"""
        try:
            depth = 1 #start with first level and deepen
            while (self.iterative == True):
                """Check time left first, and return a move immediately if critic"""
                if (self.time_left() < self.TIMER_THRESHOLD + 3):
                    #raise Timeout()
                    #print('time left: '+str(self.time_left()))
                    return next_move
                """"Perform the search for value"""
                if (self.method =="minimax"):
                    result = self.minimax(game, depth, True)
                    value = result[0]
                    new_move = result[1]
                if (self.method =="alphabeta"):
                    result = self.alphabeta(game, depth,float("-inf"),float("inf"), True)
                    value = result[0]
                    new_move = result[1]
                """Evaluate result and refresh our candidates"""
                if value > current_value:
                    current_value = value
                    next_move = new_move
                depth += 1
        except Timeout:
            return next_move
        return next_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        """Check if there are moves that are available. return otherwise"""
        if not game.get_legal_moves(self): return float('-inf'), (-1, -1)

        """Termination node: return value and position if we hit the bottom"""
        if game.is_winner(self) or game.is_loser(self):
            #print('...bottom level reached')
            return game.utility(self), game.get_player_location(self)

        """ limit for minimax drillling. use scoring as proxy value after that"""
        if depth == 0:
            scored = self.score(game, self)
            #print('scored: '+str(scored))
            return scored, game.get_player_location(self)

        """Init value and nextmove before the looping"""
        if(maximizing_player): current_value = float('-inf')
        else: current_value = float('+inf')
        next_move = (-1, -1)

        """Call minimax otherwise recursively to deepen until we reach level d deep"""
        for move in game.get_legal_moves():
            if self.time_left() < self.TIMER_THRESHOLD + 3 :
                return current_value, next_move
                #raise Timeout()
            #print('testing: '+str(move))
            """Create a copy new board for simulation"""
            newboard = game.forecast_move(move)
            value = self.minimax(newboard,  depth - 1, not maximizing_player)[0]
            """Update the value and position"""
            if((maximizing_player and (value > current_value)) or ((not maximizing_player) and (value < current_value))):
                current_value = value
                next_move = move
                #print ('>>>better move found: '+str(next_move))
        #print('best move:'+str(next_move))
        return current_value, next_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """

        """Init value and nextmove before the looping"""
        if(maximizing_player): current_value = float('-inf')
        else: current_value = float('+inf')
        next_move = (-1, -1)

        """Check if there are move that are available. return otherwise"""
        if not game.get_legal_moves(self): return float('-inf'), (-1, -1)
        #print('d:'+str(depth))

        """Termination node: we need to return value and position"""
        if game.is_winner(self) or game.is_loser(self):
            #print('...bottom level reached')
            return game.utility(self), game.get_player_location(self)

        """ limit for minimax drillling. use scoring as proxy value after that"""
        if depth == 0:
            score = self.score(game, self)
            #print('score: '+str(score))
            return score, game.get_player_location(self)

        """Call minimax otherwise recursively to deepen until we reach level d deep"""
        for move in game.get_legal_moves():
            if self.time_left() < self.TIMER_THRESHOLD + 3 :
                return current_value, next_move
                #raise Timeout()
            """Create a copy new board for simulation"""
            newboard = game.forecast_move(move)
            value = self.alphabeta(newboard,  depth - 1, alpha, beta, not maximizing_player)[0]
            """Update the value and position"""
            if((maximizing_player and (value > current_value)) or ((not maximizing_player) and (value < current_value))):
                current_value = value
                next_move = move
                #print ('>>>better move found: '+str(next_move))
            """Prune irrelevant branches now"""
            if (maximizing_player):
                if current_value >= beta: break
                if current_value > alpha: alpha = current_value
            else:
                if current_value <= alpha: break
                if current_value < beta: beta = current_value

        #print('best move:'+str(next_move))
        return current_value, next_move
