import numpy as np
import pprint
from marchMadnessBracket.utils import *
from marchMadnessBracket.Team import Team

class Bracket:
    '''
    This is a basic representation of a single elim bracket with multiples of two.
    '''
    def __init__(self,n_rounds, sim_method=higher_rated_wins):
        self.n_rounds = n_rounds
        self.n_start_teams = 2 ** n_rounds
        self.n_games = 2 ** (n_rounds ) - 1 
        self.win_loss_array = np.full(self.n_games,0)
        self.teams = np.arange( self.n_start_teams)
        self.teams = [Team(team,0.) for team in self.teams]
        self.sim_method = sim_method
    
    @classmethod
    def from_array(cls,teams, kwargs={}):
        '''
        Produces a bracket from list of teams.
        '''
        nteams = len(teams)
        n_rounds = power_of(nteams,2)
        assert n_rounds.is_integer(), "n Teams must be a power of two!"
        n_rounds = int(n_rounds)
        l_cls = cls(n_rounds, **kwargs)
        l_cls.teams = teams
        return l_cls
        
    def masked_flip(self,mask):
        '''
        Flips outcome for win_loss_array based on mask.
        '''
        assert len(mask) == self.n_games, "mask must have length of n_games"
        # check for type of mask??
        #https://stackoverflow.com/questions/36421913/apply-function-to-masked-numpy-array
        self.win_loss_array[mask] = np.logical_not(self.win_loss_array[mask])
        
    def random_flip(self,p_flip=.5):
        '''
        Randomly flip win_loss_array.
        '''
        cutoff = 1.-p_flip
        mask = np.random.rand(self.n_games) > cutoff
        self.masked_flip(mask)

    def rounds(self,sim_method=0):
        '''
        Creates bracket based on win_loss_array.
        '''
        if sim_method==0: sim_method = self.sim_method
        _current_teams = self.teams
        _win_loss_array = self.win_loss_array
        rounds = []
        for i in range(self.n_rounds):
            games = group_list(_current_teams,2)
            rounds.append(games)
            n_current = len(games)
            _this_round_wl, _win_loss_array = strip_n_elements(_win_loss_array,n_current)
            
            _current_teams = list(zip(games,_this_round_wl))
            _current_teams = list(map(lambda x: sim_method(*x[0],opposite=x[1]),_current_teams))
            
        rounds.append(_current_teams)
        return rounds
    
    def winner(self,sim_method=0):
        return self.rounds(sim_method=sim_method)[-1][0]
            
    def __repr__(self):
         return pprint.pformat(self.rounds())
         
    def __str__(self):
         return pprint.pformat(self.rounds())
        
    def round_winners(self):
        rounds = []
        for i, cur_round in enumerate(self.rounds()):
            if i == 0: continue
            round_winners = (np.array(cur_round).reshape(-1))
            round_winners
            rounds.append(list(map(lambda x: x.name, round_winners)))
        return rounds
        
    def score(self,round_winners):
        score = 0
        for r, r2 in zip(self.round_winners(),round_winners):
            matches = np.array(r)==np.array(r2)
            score+= int(np.mean(matches)*32)
        return score

    def get_chalk(self):
        return np.sum(self.win_loss_array)
        

class EvoBracket(Bracket):
    '''
    Bracket class, but it stores statistics about it's evolutionary grown. 
    '''
    def __init__(self,n_rounds, sim_method=higher_rated_wins):
        super().__init__(n_rounds, sim_method=sim_method)
        self.epoch = 0
        self.scores = []
        
        self.chalk = []
        self.wins = []
        self.resets = []

    def score(self,round_winners):
        score = super().score(round_winners)
        if len(self.scores) < self.epoch + 1:
            self.scores.append([])
        self.scores[self.epoch].append(score)
        return score

    def flatten_scores(self):
        return [np.mean(x) for x in self.scores]

    def epoch_update(self,win,reset):
        self.chalk.append(self.get_chalk())
        self.wins.append(win)
        self.resets.append(reset)
        if reset:
            self.epoch += 1