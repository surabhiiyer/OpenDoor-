from random import normalvariate, randint, randrange, sample
from collections import namedtuple
from datetime import date, timedelta
import datetime as dt
import scipy as s

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

import pylab as pl
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance

listings = [] 

Listing = namedtuple('Listing',
                    ['num_bedrooms', 'num_bathrooms', 'living_area', 'lat', 'lon',
                     'exterior_stories', 'pool', 'dwelling_type',
                     'list_date', 'list_price', 'close_date', 'close_price'])

columns = ['num_bedrooms', 'num_bathrooms', 'living_area', 'lat', 'lon','exterior_stories', 'pool', 'dwelling_type','list_date', 'list_price', 'close_date', 'close_price'] 
data_frame = pd.DataFrame(columns = columns)

DWELLING_TYPES = {'single-family', 'townhouse', 'apartment', 'patio', 'loft'}
POOL_TYPES = {'private', 'community', 'none'}


def generate_datum():
   """Returns a synthetic Listing in the Phoenix area"""
   num_bedrooms = randint(1, 4)
   num_bathrooms = randint(1, 4)
   living_area = randint(1e3, 5e3)
   list_date = random_date(date(1999, 1, 1), date(2015, 6, 1))
   list_price = randint(100e3, 500e3)
   lat = randint(33086, 33939) / float(1e3)
   lon = randint(-112649, -111437) / float(1e3)
   exterior_stories = randint(1, 3)
   pool = sample(POOL_TYPES, 1)[0]
   dwelling_type = sample(DWELLING_TYPES, 1)[0]
   is_closed = randrange(8) < 10  # 80% of listings close

   if is_closed:
       dom = randint(7, 180)
       list_to_close = normalvariate(0.03, 0.06)
       close_date = list_date + timedelta(days=dom)
       close_price = list_price * (1 - list_to_close)
   else:
       close_date = None
       close_price = None

   return Listing(num_bedrooms, num_bathrooms, living_area, lat, lon,
                  exterior_stories, pool, dwelling_type,
                  list_date, list_price, close_date, close_price)


def random_date(start_date, end_date):
   """Returns a random date between start_date and end_date"""
   delta = end_date - start_date
   return start_date + timedelta(days=randrange(delta.days))

# Executed this seperately. 
def create_dataframe():
    #create a dataframe 
    for x in range(10000): 
      l = generate_datum() 
      listings.append(l)

def query_homes(home_index):  
  # Create a Data frame with the listings. 
  create_dataframe()
  data_frame = pd.DataFrame(listings, index = range(0,10000),columns=columns)
  print "*******************************"
  print "First 5 Houses in the dataset: "
  print "*******************************"

  print data_frame[:5]
  
  #train, test = train_test_split(data_frame, test_size = 0.2)
  # select the house to check the similarity
  
  test_house = data_frame.iloc[home_index] 
  print "**********************"
  print "Test House Details: "
  print "**********************"
  
  print test_house

  dwelling_type_value = test_house['dwelling_type']

  # This can be changed, according to what you want to query. 
  grouped = data_frame.groupby('dwelling_type')

  # Create a dataframe which is grouped by a dwelling type. 
  df_townhouse = grouped.get_group(dwelling_type_value)
  
  # Use the latitude and longitude to calculate the euclidian distance. 
  # The features can be changed. Numeric values are prefered. 
  # Other features like price, living area, number of bathrooms and bedrooms can also be used 
  # To find the euclidian distance. 

  distance_columns = ['lat','lon']

  # Normalizing all the longitude and latitude valuses. There might be a large variation in distances. 
  col_numeric = df_townhouse[distance_columns] 
  df_norm = (col_numeric - col_numeric.mean())/col_numeric.std()

  # Normalized House that you will be queried against. 
  test_house_norm = df_norm.iloc[home_index] 

  # find euclidian distances between the test_house and the normalized house. 
  euclidean_distances = df_norm.apply(lambda row: distance.euclidean(row, test_house_norm), axis=1)

  # A new data frame with only the euclidian distance is created. 
  # The distances are sorted into order to pick out the top 10 houses.   
  df_euclidian = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
  df_euclidian.sort("dist",inplace=True)

  # List of top 10 similar houses. 
  ascending_dist = []
  for x in range(0,10):
    ascending_dist.append(df_euclidian.iloc[x]["idx"])

  # Print top 10 houses. 
  print "*****************"
  print "TOP TEN RESULTS: "
  print "*****************"

# Print the top 10 homes with similar listings. 
  for x in ascending_dist: 
    most_similar = data_frame.loc[int(x)] 
    print most_similar

def main():
  # This value can be changed to any number between 1 - 10000. I could have made it better by taking this input from the user. Ran out of time!  
  query_homes(4)

if __name__ == "__main__":
    main()
        