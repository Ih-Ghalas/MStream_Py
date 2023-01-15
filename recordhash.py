import numpy as np
import math as m
import sys

class Recordhash():
    def __init__(self,r,b,dim1,dim2):
        self.num_rows = r
        self.num_buckets = b
        self.dimension1 = dim1 # number of numerical features 
        self.dimension2 = dim2 # number of categorical features 
        self.num_recordhash = []
        self.cat_recordhash = []
        self.count = np.zeros((self.num_rows,self.num_buckets))
        
        '''Constructor initializes num_recordhash for numerical features and cat_recordhash 
        for categorical features and appends to a table(log_bucket, self.dimension1) elements 
        following a normal distribution, the result is of dimension (n_rows, log_bucket, dim1)'''
        log_bucket = int(np.ceil(m.log2(self.num_buckets)))
        for _ in range(self.num_rows):
            self.num_recordhash.append(np.random.randn(log_bucket, self.dimension1))
            
        '''Map integer-valued data randomly into 𝑏 buckets for the categorical features
        the result is of dimension(n_rows, dim2)'''
        self.cat_recordhash = [[0 for _ in range(self.dimension2)] for _ in range(self.num_rows)]
        for i in range(self.num_rows):
            for k in range(self.dimension2 - 1):
                self.cat_recordhash[i][k] = np.random.randint(1, self.num_buckets - 1)
            if self.dimension2:
                self.cat_recordhash[i][self.dimension2 - 1] = np.random.randint(0, self.num_buckets - 1)


    def numerichash(self,cur_numeric,i):
        """Compute the hash of a numerical record. we choose log_bucket random vectors 
        sampled from a normal distribution. We multiply each numerical feature with the
        last vector. We then map the positive scalar products to 1 and the non-positive 
        scalar products to 0 and concatenate these mapped values to get a log_bucket-bit 
        string, then convert it from a bitset into an integer 𝑏𝑢𝑐𝑘𝑒𝑡𝑛𝑢𝑚 between 
        0 and 2*log_bucket - 1"""
        log_bucket= int(np.ceil(m.log2(self.num_buckets)))
        bits= ''
        for iter in range(log_bucket):
            scalar = np.dot(self.num_recordhash[i][iter], cur_numeric)
            bits += str(int(scalar>= 0))
        return int(bits,base=2) 
    
    def categhash(self,cur_categ,i):
        """linear hash function maps each of the features into b buckets 
        then sums them and takes modulo b to have the resulting bucket 
        index of categorical hash """

        resid = 0
        for j in range(self.dimension2):
            resid = (resid + self.cat_recordhash[i][j] * cur_categ[j]) % self.num_buckets
        if resid < 0:
            resid += self.num_buckets
        return resid 
    
    def insert(self,cur_numeric,cur_categ, weight):
        """For the i-th hash and the numerical feature and categorical feature that was each 
        mapped to a bucket, we sum the resulting hash of both num and cat and apply modulo 
        to obtain the resulting bucket which is a column index of count to which we should add a weight. """
        for i in range(self.num_rows): 
            bucket1 = self.numerichash(cur_numeric, i)
            bucket2 = self.categhash(cur_categ, i)
            bucket = (bucket1 + bucket2) % self.num_buckets
            self.count[i][bucket] += weight
            
    def get_count(self,cur_numeric,cur_categ):
        """Return the minimum count (weights added in bucket) given by the different
         hashes of numerical and categ hashes"""
        min_count = sys.float_info.max
        for i in range(self.num_rows):
            bucket1 = self.numerichash(cur_numeric, i)
            bucket2 = self.categhash(cur_categ, i)
            bucket = (bucket1 + bucket2) % self.num_buckets
            min_count = min(min_count, self.count[i][bucket])
        return min_count
        
    def lower(self, factor) : 
        for i in range(self.num_rows) : 
            for j in range(self.num_buckets) : 
                self.count[i][j] = self.count[i][j] * factor
