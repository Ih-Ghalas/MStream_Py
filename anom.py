import math
import numpy as np
import numerichash
import recordhash
import categhash


def counts_to_anom(tot, cur, cur_t):
    """Computing the anomaly score."""
    cur_mean = tot / cur_t
    sqerr = max(0, cur - cur_mean)**2
    return sqerr/cur_mean + sqerr/(cur_mean * max(1, cur_t - 1))

class MStream():
    
    def __init__(self, num_rows, num_buckets, factor, dimension1, dimension2):
        self.num_rows = num_rows
        self.num_buckets = num_buckets
        self.factor = factor
        self.dimension1 = dimension1
        self.dimension2 = dimension2
        self.cur_t = 1

        self.cur_count = recordhash.Recordhash(num_rows, num_buckets, dimension1, dimension2)
        self.total_count = recordhash.Recordhash(num_rows, num_buckets, dimension1, dimension2)

        self.numeric_score = [numerichash.Numerichash(num_rows, num_buckets) for i in range(dimension1)]
        self.numeric_total = [numerichash.Numerichash(num_rows, num_buckets) for i in range(dimension1)]
        self.categ_score = [categhash.Categhash(num_rows, num_buckets) for i in range(dimension2)]
        self.categ_total = [categhash.Categhash(num_rows, num_buckets) for i in range(dimension2)]

        if dimension1 > 0 : 
            self.max_numeric = [-float('inf') for i in range(dimension1)]
            self.min_numeric = [float('inf') for i in range(dimension1)]


    def learn_one(self, numeric, categ, times):
        
        if (times > self.cur_t):
            self.cur_count.lower(self.factor)
            for j in range(self.dimension1):
                self.numeric_score[j].lower(self.factor)
            for j in range(self.dimension2):
                self.categ_score[j].lower(self.factor)
            self.cur_t = times

        cur_numeric = numeric
        cur_categ = categ

        for node_iter in range(self.dimension1):
            cur_numeric[node_iter] = np.log10(1 + cur_numeric[node_iter])
            self.min_numeric[node_iter] = min(self.min_numeric[node_iter], cur_numeric[node_iter])
            self.max_numeric[node_iter] = max(self.max_numeric[node_iter], cur_numeric[node_iter])

            if (self.max_numeric[node_iter] == self.min_numeric[node_iter]):
                cur_numeric[node_iter] = 0
            else:
                cur_numeric[node_iter] = (cur_numeric[node_iter] - self.min_numeric[node_iter])/\
                    (self.max_numeric[node_iter] - self.min_numeric[node_iter])
            
            self.numeric_score[node_iter].insert(cur_numeric[node_iter], 1)
            self.numeric_total[node_iter].insert(cur_numeric[node_iter], 1)

        self.cur_count.insert(cur_numeric, cur_categ, 1)
        self.total_count.insert(cur_numeric, cur_categ, 1)

        for node_iter in range(self.dimension2):
            self.categ_score[node_iter].insert(cur_categ[node_iter], 1)
            self.categ_total[node_iter].insert(cur_categ[node_iter], 1)

        return(self) 
    

    def score_one(self, numeric, categ, times) : 
        #check type of times
      
        cur_numeric = numeric
        cur_categ = categ
         
        if (times > self.cur_t):
            self.cur_count.lower(self.factor)
            for j in range(self.dimension1):
                self.numeric_score[j].lower(self.factor)
            for j in range(self.dimension2):
                self.categ_score[j].lower(self.factor)
            self.cur_t = times
   
            
        sum = 0
        for node_iter in range(self.dimension1): 
            cur_numeric[node_iter] = math.log10(1 + cur_numeric[node_iter])
            
            self.min_numeric[node_iter] = min(self.min_numeric[node_iter], cur_numeric[node_iter])
            self.max_numeric[node_iter] = max(self.max_numeric[node_iter], cur_numeric[node_iter])
            if (self.max_numeric[node_iter] == self.min_numeric[node_iter]) :
                cur_numeric[node_iter] = 0
            else :
                cur_numeric[node_iter] = (cur_numeric[node_iter] - self.min_numeric[node_iter]) /\
                (self.max_numeric[node_iter] - self.min_numeric[node_iter])
            
            
            self.numeric_score[node_iter].insert(cur_numeric[node_iter],1)
            self.numeric_total[node_iter].insert(cur_numeric[node_iter],1)
            
            t = counts_to_anom(self.numeric_total[node_iter].get_count(cur_numeric[node_iter]),
                            self.numeric_score[node_iter].get_count(cur_numeric[node_iter]), self.cur_t)
            sum = sum+t

        self.cur_count.insert(cur_numeric, cur_categ, 1)
        self.total_count.insert(cur_numeric, cur_categ, 1)

        for node_iter in range(self.dimension2): 
            self.categ_score[node_iter].insert(cur_categ[node_iter], 1)
            self.categ_total[node_iter].insert(cur_categ[node_iter], 1)
                                                                     
            t = counts_to_anom(self.categ_total[node_iter].get_count(cur_categ[node_iter]),
                                self.categ_score[node_iter].get_count(cur_categ[node_iter]), self.cur_t)
            sum = sum+t
        
        cur_score = counts_to_anom(self.total_count.get_count(cur_numeric, cur_categ),
                                    self.cur_count.get_count(cur_numeric, cur_categ), self.cur_t)
            
        sum = sum + cur_score
        anom_score = math.log(1 + sum)
        return anom_score
