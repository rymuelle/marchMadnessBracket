
class Team:
    def __init__(self,name,rating=0):
        self.name = name
        self.rating = rating
    def __repr__(self):
         return "Team:({},{:.2f})".format(self.name,self.rating)
    def __str__(self):
         return "{},{:.2f}".format(self.name,self.rating)