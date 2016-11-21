# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Read files from local storge and turn into DF
train_file = "C:\\Users\\Justin\\Development\\Kaggle_Titanic\\data\\train.csv"
test_file = "C:\\Users\\Justin\\Development\\Kaggle_Titanic\\data\\test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)


####Creat Function for cleaning dataframes, repeatable on both Training and Test DF ###
def clean_dataframe(df):
	########  Beg - Data Cleaning  - Clean existing columns and create new columns#####################
	###### Turning Categorical to integers######
	#### Sex 
	df['Sex_Integer'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

	###### Fill Missing Values ###################
	#### Age
	# Create  and fill an array that will split passengers on an array of Sex and Passenger Class
	median_ages = np.zeros((2,3))
	for i in range(0,2):
		for j in range(0,3):
			median_ages[i,j] = df[(df['Sex_Integer'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()	
	# Fill "Age Fill" based on "Age"  & Passenger Class
	df['Age_Fill'] = df['Age']
	for i in range(0, 2):
	    for j in range(0, 3):
	        df.loc[ (df.Age.isnull()) & (df.Sex_Integer == i) & (df.Pclass == j+1),'Age_Fill'] = median_ages[i,j]
	#### AgeIsNull
	df['Age_Is_Null'] = pd.isnull(df.Age).astype(int)

	# Create a column Fare_Fill that fills in fare Data that is null (set to Median Fare in PClass)
	number_of_classes = len(df['Pclass'].unique())
	median_fares_by_class = np.zeros([number_of_classes],float)
	
	for l in range(0,3):
		median_fares_by_class[l] = df[(df['Pclass'] == l+1)]['Fare'].dropna().median()
		
	df['Fare_Fill'] = df['Fare']
	for k in range(0,3):
		df.loc[(df.Fare.isnull()) & (df.Pclass == k+1),'Fare_Fill' ] = median_fares_by_class[k]

	#### Fare IS Null
	df['Fare_Is_Null'] = pd.isnull(df.Fare).astype(int)

	### Embarked Filling the empties 
	df['Embarked_Fill'] = df['Embarked']
	most_prevalent_embarked = df.Embarked.value_counts(sort=True, ascending = False).idxmax()
	df.loc[ df.Embarked.isnull(),'Embarked_Fill'] = most_prevalent_embarked
	df['Embarked_Fill_Integer'] = df['Embarked_Fill'].map( {'S': 0, 'C': 1,'Q': 2} ).astype(float)


	#########  End - Data Cleaning  - Clean existing columns and create new columns#####################

	##########  Beg - Feature Engineering - Create new variables from existing ones @########################
	### Family Size
	df['Family_Size'] = df['SibSp'] + df['Parch']
	### Age * Class
	df['Age_x_Class'] = df.Age_Fill * df.Pclass
	###### Future Possible Features to Engineer: ###
	#### Names (Titles For men.... (does Honorary title predict survival?)
	### Ticket? 
	#### Cabin?
	##################  End - Feature Engineering - Create new variables from existing ones @########################

	return df


# Run Functions that clean up the dataframes 
train_df = clean_dataframe(train_df)
test_df = clean_dataframe(test_df)



############# Beg =  Convert Data into format ready for Machine Learning ###############

#### Determine existing data type - Looking to through away string
# Print all Datatype
# print train_df.dtypes
# Print only  Datatype that are objects 
# print train_df.dtypes[train_df.dtypes.map(lambda x: x=='object')]

### Drop Data that are not Needed
# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values

# dropping variables 
train_df = train_df.drop(['PassengerId','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age','Fare','Embarked_Fill'], axis=1) 
test_df = test_df.drop(['PassengerId','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age','Fare','Embarked_Fill'], axis=1) 

print test_df.dtypes

# Begin Making prediction on Test Data using simple algorithms
# Uncomment to fill in a prediction on test_dataframe
# # All Die
# test_df['Survived_All_Dead'] = 0 
# # All Alive
# test_df['Survived_All_Alive'] = 1
# # All Females Live
# test_df['Survived_All_Females_Live'] = np.where(test_df['Sex_Integer'] == 1, 1, 0)





# ####################################################### Beg = Gender  Class and Price Bracket ###########################################################################################
# #### Define  Fare Ceiling and Price Bracket size 
# fare_ceiling = 40
# fare_bracket_size = 10
# number_of_price_brackets = fare_ceiling/fare_bracket_size
# number_of_classes = len(train_df['Pclass'].unique())

# ###### TRAINING DATA# ############
# # Create a column called "Fare_Transformed" that caps fairs over the fair ceiling 
# train_df['Fare_Transformed'] =  np.where(train_df['Fare'] > fare_ceiling, fare_ceiling-1,train_df['Fare'] )
# ##### Initialize Fare_Bin
# train_df['Fare_Bin'] =  -1
# ### Iterate through the price brackets and update Fare_Bine based on Fare_transformed
# ### Note THis isn't perfect Hard coding the value of  could adapt this in ideal world, for time saving use hardcoded value
# train_df['Fare_Bin'] = np.where(train_df.Fare_Transformed<10, 1,np.where(train_df.Fare_Transformed<20, 2,np.where(train_df.Fare_Transformed<30, 3,np.where(train_df.Fare_Transformed<40, 4,-1))))


# ###### TEST DATA# ############
# # Create a column called "Fare_Transformed" that caps fairs over the fair ceiling 
# test_df['Fare_Transformed'] =  np.where(test_df['Fare'] > fare_ceiling, fare_ceiling-1,test_df['Fare'] )
# ##### Initialize Fare_Bin
# test_df['Fare_Bin'] =  -1
# ### Iterate through the price brackets and update Fare_Bine based on Fare_transformed
# ### Note THis isn't perfect Hard coding the value of  could adapt this in ideal world, for time saving use hardcoded value
# test_df['Fare_Bin'] = np.where(test_df.Fare_Transformed<10, 1,np.where(test_df.Fare_Transformed<20, 2,np.where(test_df.Fare_Transformed<30, 3,np.where(test_df.Fare_Transformed<40, 4,-1))))



# ## Initialize a Survival Table holding summary survivals rate for gender, class, price ticket bin 
# survival_table = np.zeros([2,number_of_classes,number_of_price_brackets],float)


# ### Turn Dataframe int an arrays for Machine Learning algorithms
# train_data = train_df.values
# test_data = test_df.values

# ##### Cycle through the data set and store (into survival_table) the survival rates for  Gender, Class, and Price bracket
# for i in xrange(number_of_classes):
#     for j in xrange(number_of_price_brackets):

#         women_only_stats = train_data[ (train_data[0::,6] == 0) \
#                                  & (train_data[0::,2].astype(np.float) == i+1) \
#                                  & (train_data[0:,12].astype(np.float) >= j*fare_bracket_size) \
#                                  & (train_data[0:,13].astype(np.float) < (j+1)*fare_bracket_size), 1]

#         men_only_stats = train_data[ (train_data[0::,6] != 0) \
#                                  & (train_data[0::,2].astype(np.float) == i+1) \
#                                  & (train_data[0:,12].astype(np.float) >= j*fare_bracket_size) \
#                                  & (train_data[0:,13].astype(np.float) < (j+1)*fare_bracket_size), 1]

#                                  #if i == 0 and j == 3:

#         survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))  # Female stats
#         survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))    # Male stats


# #### Update the probabilities to be either 0  or 1 
# survival_table[ survival_table < 0.5 ] = 0
# survival_table[ survival_table >= 0.5 ] = 1 


# # Apply Survival Table to Test Data 

# # Now I have my indicator I can read in the test file and write out

# test_file = open('..\\data\\test.csv', 'rb')
# test_file_object = csv.reader(test_file)
# header = test_file_object.next()

# # print test_file_object

# # Also open the a new file so I can write to it. 
# predictions_file = open("..\\output\\predictions_gender_class_pricebin.csv", "wb")
# predictions_file_object = csv.writer(predictions_file)
# predictions_file_object.writerow(["PassengerId", "Survived"])


# # First thing to do is bin up the price file
# for row in test_file_object:
#     for j in xrange(number_of_price_brackets):
#         # If there is no fare then place the price of the ticket according to class
#         try:
#             row[8] = float(row[8])    # No fare recorded will come up as a string so
#                                       # try to make it a float
#         except:                       # If fails then just bin the fare according to the class
#             bin_fare = 3 - float(row[1])
#             break                     # Break from the loop and move to the next row
#         if row[8] > fare_ceiling:     # Otherwise now test to see if it is higher
#                                       # than the fare ceiling we set earlier
#             bin_fare = number_of_price_brackets - 1
#             break                     # And then break to the next row

#         if row[8] >= j*fare_bracket_size\
#             and row[8] < (j+1)*fare_bracket_size:     # If passed these tests then loop through
#                                                       # each bin until you find the right one
#                                                       # append it to the bin_fare
#                                                       # and move to the next loop
#             bin_fare = j
#             break
#         # Now I have the binned fare, passenger class, and whether female or male, we can
#         # just cross ref their details with our survival table
#     if row[3] == 'female':
#         predictions_file_object.writerow([row[0], "%d" % int(survival_table[ 0, float(row[1]) - 1, bin_fare ])])
#     else:
#         predictions_file_object.writerow([row[0], "%d" % int(survival_table[ 1, float(row[1]) - 1, bin_fare])])

# # # Close out the files
# test_file.close()
# predictions_file.close()


# ####################################################### End =  Gender  Class and Price Bracket ###########################################################################################








############# Beg =  Convert Data into format ready for Machine Learning ###############

# ### Turn Dataframe int an arrays for Machine Learning algorithms
train_data = train_df.values
test_data = test_df.values


#### Run Random Forrest 

# Create the random forest object which will include all the parameters for the fit
print "Training..."
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

print "Predicting..."
output = forest.predict(test_data).astype(int)



################# Beg  Basic Stats  ############
# train_df.info()
# print train_df.describe()
# print train_df['Age'].mean()
# print train_df['Age'].median()
# print train_df[train_df['Age']>60].mean()
# Plot subsets
# print train_df[train_df['Age'].isnull()][['Sex', 'Pclass', 'Age']]



########## plot histograms
### Age
# # Note Will through an error because of NaNs
# plt.hist(train_df['Age'],bins=16, range=(0,100), alpha = .5)
# plt.title("Titanic Passengers: Age Histogram")
# plt.xlabel("Age of Passengers")
# plt.ylabel("Frequency of Occurrence")
# plt.show()
# ### Age_Fill
# plt.hist(train_df['Age_Fill'],bins=16, range=(0,100), alpha = .5)
# plt.title("Titanic Passengers: Age_Fill Histogram")
# plt.xlabel("Age of Passengers")
# plt.ylabel("Frequency of Occurrence")
# plt.show()

# ### Family_Size
# plt.hist(train_df['Family_Size'],bins=10, range=(0,10), alpha = .5)
# plt.title("Titanic Passengers: Family_Size Histogram")
# plt.xlabel("Family Size")
# plt.ylabel("Frequency of Occurrence")
# plt.show()

# ### Age X Class
# plt.hist(train_df['Age_x_Class'],bins=20, alpha = .5)
# plt.title("Titanic Passengers: Age_x_Class Histogram")
# plt.xlabel("Age_x_Class")
# plt.ylabel("Frequency of Occurrence")
# plt.show()






###### Prepare a Predictions File  using Pandas Dataframe  ###########

# # initilize df
# predictions_df = pd.DataFrame({})
# # Copy over Passenger ID
# predictions_df['PassengerId'] = test_df['PassengerId']
# # Set the Passenger ID as the Index 
# predictions_df = predictions_df.set_index('PassengerId')


########### Prepare predication File - Using CSV Writer ############

predictions_file = open("..\\output\\predictions_random_forest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'





### Pick a method for determining alive or dead and then write the prediction fie

##  All Die 
# predictions_df['Survived'] = test_df['Survived_All_Dead']
# prediction_file = "C:\\Users\\Justin\\Development\\Kaggle_Titanic\\output\\predictions_all_dead.csv"
# predictions_df.to_csv(prediction_file,sep=',')

# All LIve
# predictions_df['Survived'] = test_df['Survived_All_Alive']
# prediction_file = "C:\\Users\\Justin\\Development\\Kaggle_Titanic\\output\\predictions_all_alive.csv"
# predictions_df.to_csv(prediction_file,sep=',')

# All Females Live
# predictions_df['Survived'] = test_df['Survived_All_Females_Live']
# prediction_file = "C:\\Users\\Justin\\Development\\Kaggle_Titanic\\output\\predictions_all_females_alive.csv"
# predictions_df.to_csv(prediction_file,sep=',')


# ### Exporting the Training Dataframe
# train_data_file = "C:\\Users\\Justin\\Development\\Kaggle_Titanic\\output\\train_data_cleaned.csv"
# train_df.to_csv(train_data_file,sep=',')


# ### Exporting the Test Dataframe
# test_data_file = "C:\\Users\\Justin\\Development\\Kaggle_Titanic\\output\\test_data_cleaned.csv"
# test_df.to_csv(test_data_file,sep=',')






# Test Printing
# print train_df.head()
# print test_df.head()
# print train_df
# print test_df
# print median_ages
# print train_df[ train_df['Age'].isnull() ][['Sex_Integer','Pclass','Age','Age_Fill']].head(10)   # looking for N/As in Age
# print train_df[ train_df['Fare'].isnull() ][['Sex_Integer','Pclass','Age_Fill']].head(10)   # looking for N/As in Age
# print test_df[ test_df['Fare'].isnull() ][['Sex_Integer','Pclass','Age_Fill']].head(10)   # looking for N/As in Age

# print train_data
# print predictions_df.head()





