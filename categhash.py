import numpy as np 
import sys


class Categhash :
    def __init__(self, r, b) :
        #initializing parameters
        self.num_rows= r
        self.num_buckets = b

        #creating the hash table 
        self.hash_a = np.zeros(self.num_rows)
        self.hash_b = np.zeros(self.num_rows)

        #initializing the count 
        self.count = np.zeros((self.num_rows, self.num_buckets))

        #initialize the elements of hash_a and hash_b betwen 0 to b-1 ( i hashes )
        for i in range(self.num_rows) :
            self.hash_a[i] = np.random.randint(self.num_buckets - 1) + 1   
            self.hash_b[i] = np.random.randint(self.num_buckets)  	       

    def hash(self, a, i) :
        """defining the i-th hash function that takes the feature and multiply it by 
        the i-th element of hash_a and add the i-th element of hash_b
        we take it modulo num_bucket so that it can be mapped to one of the b buckets"""
        resid = (a * self.hash_a[i] + self.hash_b[i]) % self.num_buckets   
        if resid < 0 : 
            return int(resid) + self.num_buckets
        return int(resid )

    def insert( self, cur_int,  weight):
        """For the i ème hash and the current feature(that was mapped to a bucket), 
        we add a weight in table count  """
        for i in range(self.num_rows) :
            bucket = self.hash(cur_int, i)
            self.count[i][bucket] += weight
                        
    
    def get_count(self, cur_int) :
        """Return the minimum count (weights added in bucket) given by the different 
        hashes of cur_int """
        min_count = sys.float_info.max
        for i in range(self.num_rows):
            bucket = self.hash(cur_int, i)
            min_count = min(min_count, self.count[i][bucket])    
        return min_count 
        
    def lower(self, factor) : 
        """multiply the element of count by a factor to lower the values in the table"""
        for i in range(self.num_rows) : 
            for j in range(self.num_buckets) : 
                self.count[i][j] = self.count[i][j] * factor

    def clear(self) : 
        """reset to zero"""
        self.count = np.zeros((self.num_rows, self.num_buckets))
