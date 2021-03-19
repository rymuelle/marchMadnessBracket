import torch
from math import e
import math
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from marchMadnessBracket.Bracket import Bracket
from marchMadnessBracket.utils import realistic_game_logisitic
from marchMadnessBracket.utils import *

def logisitic(x,y,prob1,prob2,scale=9.5):
    '''This function takes two lists of powers and probs and returns a combined prob.
       We use the logisitic function because it is analyitical, while the normal cdf is not.
       e*2 makes a good approximation of the normal cdf however.'''
    delta = x.view(1,-1)-y.view(1,-1).T
    p_matrix = (1/(1+torch.pow(e*2, (-(delta)/scale) ) ))
    prob2 = (prob2.view(1,-1).T)
    prob1 = (prob1.view(1,-1))
    ones = torch.full((2,2),1)
    x1p = (p_matrix*prob1*prob2).sum(dim=0)
    x2p = ((1-p_matrix)*prob1*prob2).sum(dim=1)
    return torch.cat( (x1p,x2p)).view(-1,2)
 
def comb(x,y,scale=9.5):
    '''This function just combines and formats the view.'''
    return torch.cat((x,y)).view(-1,2)

def one_round(powers, prev_round_prob, view_dim,split_dim,function):
    '''simulates one round of play and returns output probability.'''
    next_round = torch.Tensor()
    n_rounds = 0
    for power,prob in zip(powers.view(-1,view_dim),prev_round_prob.view(-1,view_dim)):
        pow1,pow2 = power.split(split_dim)
        prob1,prob2 = prob.split(split_dim)
        p_mat = logisitic(pow1,pow2,prob1,prob2)
        next_round = torch.cat( (next_round,p_mat) )
        n_rounds+=1
    return next_round.view(-1), n_rounds

def do_full_bracket(rankings,prob):
    power = math.log(rankings.shape[0])/math.log(2)
    assert power.is_integer(), "must be multiple of two"
    power = int(power)
    for x in range(power):
        l_power = 2**x
        prob,n_rounds = one_round(rankings,prob,2*l_power,l_power,logisitic)
    return prob

def do_full_bracket_multi_level(rankings,prob):
    probs = torch.Tensor()
    power = math.log(rankings.shape[0])/math.log(2)
    assert power.is_integer(), "must be multiple of two"
    power = int(power)
    for x in range(power):
        l_power = 2**x
        prob,n_rounds = one_round(rankings,prob,2*l_power,l_power,logisitic)
        probs = torch.cat( (probs,prob) )
        #print(prob)
    return probs

class random_vector:
    def __init__(self,lenght, kwargs={}):
        self.input_tensor = torch.full((1, 4), 1., requires_grad=True)
        self.bl = BayesianLinear(4,lenght, **kwargs)
        self.bl2 = BayesianLinear(lenght,lenght, **kwargs)
    def roll(self):
        output = self.bl(self.input_tensor)
        return self.bl2(output)
    
def make_target_from_human_picks(human_picks,names):
    targets = [[human_picks[name][x]/100. for name in names] for x in range(6)]
    target = []
    for x in targets:
        target = target+x
    return torch.tensor(target)

class TorchBracket(Bracket):
    def __init__(self,teams, sim_method=realistic_game_logisitic):
        nteams = len(teams)
        n_rounds = power_of(nteams,2)
        assert n_rounds.is_integer(), "n Teams must be a power of two!"
        n_rounds = int(n_rounds)
        super().__init__(n_rounds, sim_method=sim_method)
        self.teams = teams    
        self.ratings_tensor = torch.tensor([team.rating for team in self.teams], requires_grad=True)
        self.starting_prob = torch.full((1,64),1.)
        for team in teams:
            team.starting_rating = team.rating

    def prob_multi_level_flat(self):
        return do_full_bracket_multi_level(self.ratings_tensor,self.starting_prob)
    
    def prob_multi_level(self):
        prob = self.prob_multi_level_flat()
        return prob.view(self.n_rounds,-1)

    def numpy_rating_tensor(self):
        return self.ratings_tensor.detach().numpy()
    
    def numpy_round_prob(self):
        return self.ratings_tensor.detach().numpy()
    
    def set_teams(self):
        np_ratings = self.numpy_rating_tensor()
        probs = self.prob_multi_level()
        round_probs = probs.T.detach().numpy()
        for rating,team,round_prob in zip(np_ratings, self.teams,round_probs):
            team.rating = rating
            team.round_prob = round_prob
    
    def trained_rating(self):
        return np.array([team.rating for team in self.teams])
    
    def starting_rating(self):
        return np.array([team.starting_rating for team in self.teams])