import numpy as np
import math 

def power_of(num,power):
    return math.log(num)/math.log(power)

def strip_n_elements(l_array,n_ele):
    return l_array[:n_ele],l_array[n_ele:]

def group_list(l_group,n_grouping):
    half_size = len(l_group)/n_grouping
    assert half_size.is_integer(), "list must be divisible by {}".format(n_grouping)
    half_size = int(half_size)
    return [[l_group[2*i],l_group[2*i+1]] for i in range(half_size)]

def higher_rated_wins(team1,team2,opposite=0):
    if (team1.rating - team2.rating)*(.5-opposite) > 0: team = team1
    else: team = team2
    return team

def realstic_game(team1,team2,opposite=0, width=9.5):
    from scipy.stats import norm
    win_prob = norm.cdf(team1.rating-team2.rating, loc=0, scale=width)*(.5-opposite)*2
    random = np.random.rand()*(.5-opposite)*2
    if random < win_prob and not opposite: return team1
    else: return team2

def realistic_game_logisitic(team1,team2,opposite=0, width=9.5):
    from scipy.stats import norm
    win_prob = 1./(1+(math.e*2)**(-(team1.rating-team2.rating)/width))
    random = np.random.rand()*(.5-opposite)*2
    if random < win_prob and not opposite: return team1
    else: return team2

def color_fader(c1,c2,steps): 
    '''fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)'''
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    mixes = np.linspace(0,1,steps)
    return [mpl.colors.to_hex((1-mix)*c1 + mix*c2)  for mix in mixes]

def r_squared(y,f):
    s_tot = np.sum((y-np.mean(y))**2)
    res = np.sum((y-f)**2)
    return (1-res/s_tot)