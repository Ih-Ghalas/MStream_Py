import math
import numpy as np 

class Numerichash : 
    """Mapping the features to buckets"""
    def __init__(self, r,b) : 
        """initialize the parameters """
        self.num_rows = r
        self.num_buckets = b
        self.count = np.zeros((self.num_rows, self.num_buckets))

    def hash(self,cur_node) :
        """Defining the hash function : we multiply the feature by the number 
        of buckets and then we take the integer of the result"""
        cur_node = cur_node * (self.num_buckets - 1)
        bucket = math.floor(cur_node)
        if(bucket < 0) :
            bucket = (bucket%self.num_buckets + self.num_buckets)%self.num_buckets
        return bucket

    def insert(self, cur_node,  weight) : 
        """Add weight in table count for the index bucket to which the feature was mapped."""
        bucket = self.hash(cur_node)
        self.count[0][bucket] += weight

    def get_count(self, cur_node) :
        """Return the weight of cur_node that resulted from mapping it to the bucket."""
        bucket = self.hash(cur_node)
        return self.count[0][bucket]

    def lower(self, factor) :
        """Multiply the element of count by a factor to lower the values in the table."""
        for i in range(0,self.num_rows) :
            for j in range(0,self.num_buckets):
                self.count[i][j] = self.count[i][j] * factor
    
    def clear(self) :
        """Reset to zero."""
        self.count = np.zeros((self.num_rows, self.num_buckets))
