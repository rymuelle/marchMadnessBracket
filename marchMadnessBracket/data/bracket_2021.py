from marchMadnessBracket.utils import *
from marchMadnessBracket.Team import Team
from marchMadnessBracket.Bracket import Bracket
teams = [['Gonzaga', 25.9],
 ['NORF/APP', 4.2],
 ['Oklahoma', 14.1],
 ['Missouri', 11.4],
 ['Creighton', 16.9],
 ['UCSB', 6.4],
 ['Virginia', 16.2],
 ['Ohio', 5.2],
 ['USC', 15.9],
 ['WICH/DRKE', 8.4],
 ['Kansas', 17.1],
 ['Eastern Wash.', 4.6],
 ['Oregon', 13.9],
 ['VCU', 10.7],
 ['Iowa', 20.9],
 ['Grand Canyon', 4.5],
 ['Michigan', 20.4],
 ['MSM/TXSO', 4.0],
 ['LSU', 14.6],
 ['St. Bonaventure', 12.7],
 ['Colorado', 16.0],
 ['Georgetown', 9.7],
 ['Florida St.', 16.7],
 ['N.C. Greensboro', 5.1],
 ['BYU', 13.9],
 ['MSU/UCLA', 12.6],
 ['Texas', 16.5],
 ['Abilene Christian', 6.0],
 ['Connecticut', 15.2],
 ['Maryland', 12.6],
 ['Alabama', 18.0],
 ['Iona', 0.1],
 ['Baylor', 23.1],
 ['Hartford', 2.4],
 ['N. Carolina', 14.9],
 ['Wisconsin', 15.9],
 ['Villanova', 17.9],
 ['Winthrop', 5.8],
 ['Purdue', 15.2],
 ['North Texas', 7.9],
 ['Texas Tech', 16.9],
 ['Utah St.', 12.1],
 ['Arkansas', 16.2],
 ['Colgate', 8.8],
 ['Florida', 13.2],
 ['Virginia Tech', 12.3],
 ['Ohio St.', 17.5],
 ['Oral Roberts', 0.9],
 ['Illinois', 21.7],
 ['Drexel', 0.8],
 ['Loyola Chicago', 13.9],
 ['Georgia Tech', 13.1],
 ['Tennessee', 16.5],
 ['Oregon St.', 7.8],
 ['Oklahoma St.', 14.3],
 ['Liberty', 4.6],
 ['SDSU', 14.5],
 ['Syracuse', 12.3],
 ['West Virginia', 16.2],
 ['Morehead St.', 0.1],
 ['Clemson', 12.5],
 ['Rutgers', 13.0],
 ['Houston', 19.9],
 ['Cleveland St.', 2.0]]

teams = list(map(lambda x: Team(*x), teams))


human_picks = {'Gonzaga': [98.9519, 94.8757, 87.798, 74.4417, 62.1272, 44.7553],
 'Michigan': [97.7703, 86.4404, 67.3721, 45.5894, 13.9436, 8.3917],
 'Baylor': [97.3671, 84.6983, 72.1768, 51.5311, 27.0272, 8.4005],
 'Illinois': [96.9702, 90.0567, 75.9362, 60.3624, 38.611, 17.0228],
 'Ohio St.': [96.7406, 84.1928, 62.079, 25.6382, 10.1104, 3.0637],
 'Iowa': [96.7113, 79.8003, 51.4085, 10.5339, 5.4266, 2.298],
 'Houston': [95.3125, 79.7777, 47.0808, 14.1868, 6.0026, 1.7483],
 'Kansas': [95.2037, 71.6401, 33.1194, 7.6329, 3.8451, 1.9184],
 'Alabama': [94.4231, 77.613, 46.7196, 21.6327, 4.6305, 1.9123],
 'Texas': [94.2555, 73.5456, 36.5782, 15.2911, 2.8173, 1.0344],
 'West Virginia': [92.9193, 66.7596, 34.9776, 8.8696, 3.2486, 0.8596],
 'Florida St.': [91.9959, 64.3256, 20.4014, 8.0377, 1.4646, 0.5707],
 'Purdue': [90.8554, 49.6968, 8.7982, 3.2559, 1.0908, 0.3848],
 'Arkansas': [87.4789, 56.0917, 19.0892, 5.6129, 1.8212, 0.5172],
 'Oklahoma St.': [86.4623, 57.9704, 11.9466, 7.0215, 2.8562, 0.785],
 'Virginia': [81.7177, 54.7345, 6.3997, 2.9869, 1.2928, 0.6111],
 'USC': [79.9809, 23.1631, 6.6405, 0.7899, 0.3104, 0.149],
 'Tennessee': [78.3385, 32.7545, 6.1158, 2.379, 0.785, 0.2583],
 'Texas Tech': [74.3056, 33.74, 10.1473, 2.2028, 0.6679, 0.2311],
 'Villanova': [73.2305, 40.8853, 9.8038, 5.0068, 2.5424, 0.7539],
 'Creighton': [72.5547, 30.7618, 1.9908, 0.6822, 0.2813, 0.1222],
 'Oregon': [68.0303, 14.9725, 5.9661, 0.8905, 0.3805, 0.1845],
 'LSU': [66.0674, 9.0171, 3.6814, 1.1521, 0.3042, 0.1421],
 'BYU': [62.9535, 15.7053, 5.0334, 1.0696, 0.23, 0.1029],
 'Oklahoma': [59.7805, 2.5549, 1.4619, 0.4152, 0.1608, 0.0759],
 'Clemson': [58.1012, 11.3748, 3.804, 1.0717, 0.3676, 0.1267],
 'SDSU': [55.8794, 19.715, 7.0245, 1.292, 0.4584, 0.1827],
 'Loyola Chicago': [52.6569, 4.436, 1.8878, 0.7611, 0.3015, 0.1276],
 'N. Carolina': [52.3831, 8.1146, 4.5927, 2.4464, 1.1705, 0.5811],
 'Colorado': [51.3696, 18.2313, 3.4298, 1.172, 0.2592, 0.1152],
 'Connecticut': [50.3559, 11.0473, 4.539, 1.5877, 0.4821, 0.2954],
 'Florida': [50.2091, 7.9685, 3.4547, 0.8888, 0.3954, 0.166],
 'Virginia Tech': [47.6214, 5.6684, 2.1198, 0.3559, 0.14, 0.0576],
 'Maryland': [47.4399, 8.1689, 2.4206, 0.4912, 0.1539, 0.0793],
 'Georgetown': [46.5752, 14.3419, 2.5434, 1.0835, 0.4038, 0.2642],
 'Wisconsin': [45.7251, 5.3488, 2.4313, 1.048, 0.3634, 0.1805],
 'Georgia Tech': [44.9111, 3.3189, 1.3556, 0.5858, 0.2266, 0.1064],
 'Syracuse': [41.6842, 10.3958, 3.5551, 1.2216, 0.6556, 0.4539],
 'Rutgers': [39.4335, 6.1392, 1.3447, 0.2788, 0.1109, 0.059],
 'Missouri': [38.9081, 1.6151, 0.6383, 0.27, 0.1148, 0.0617],
 'MSU/UCLA': [34.5499, 7.7077, 2.1479, 0.6696, 0.2409, 0.1683],
 'St. Bonaventure': [31.8321, 2.6438, 0.7528, 0.2623, 0.0827, 0.0365],
 'VCU': [30.5645, 3.3342, 0.8051, 0.1448, 0.0538, 0.0238],
 'UCSB': [26.1946, 6.0954, 0.3257, 0.1323, 0.0565, 0.0292],
 'Winthrop': [24.9359, 6.2573, 0.4682, 0.1861, 0.0713, 0.0259],
 'Utah St.': [23.6008, 5.0369, 0.7673, 0.1793, 0.0685, 0.0279],
 'Oregon St.': [19.2498, 3.6529, 0.57, 0.2303, 0.1141, 0.049],
 'WICH/DRKE': [18.3386, 2.9285, 0.5167, 0.1193, 0.0517, 0.0249],
 'Ohio': [17.1515, 7.2972, 0.7091, 0.2074, 0.0868, 0.0347],
 'Liberty': [11.2466, 3.7612, 0.4992, 0.1821, 0.0735, 0.0345],
 'Colgate': [10.6518, 3.5124, 0.791, 0.2841, 0.1674, 0.0327],
 'North Texas': [7.347, 1.6239, 0.3092, 0.1158, 0.048, 0.0222],
 'N.C. Greensboro': [6.0465, 1.4522, 0.2921, 0.1147, 0.046, 0.0174],
 'Morehead St.': [4.8834, 1.2814, 0.3477, 0.127, 0.0574, 0.0252],
 'Iona': [4.0037, 1.6154, 0.761, 0.4502, 0.3247, 0.0284],
 'Abilene Christian': [3.9757, 1.2719, 0.3811, 0.1638, 0.0562, 0.025],
 'Eastern Wash.': [3.8335, 1.1078, 0.2809, 0.0934, 0.0419, 0.0213],
 'Cleveland St.': [2.8107, 0.9433, 0.3745, 0.1365, 0.0581, 0.0279],
 'Grand Canyon': [2.5684, 0.9436, 0.4126, 0.1492, 0.0677, 0.0383],
 'Oral Roberts': [1.83, 0.6828, 0.2949, 0.1078, 0.0448, 0.0224],
 'Hartford': [1.5213, 0.502, 0.2671, 0.1587, 0.0838, 0.0442],
 'Drexel': [1.4445, 0.5943, 0.3273, 0.1886, 0.0881, 0.0472],
 'MSM/TXSO': [0.9141, 0.396, 0.2122, 0.1282, 0.0524, 0.0184],
 'NORF/APP': [0.8321, 0.3442, 0.1907, 0.1149, 0.056, 0.0234]}