
import numpy as np 
import pandas as pd 
import sys
import pdb
import matplotlib.pylab as plt
from sklearn import preprocessing as sp
import  scipy.spatial.distance as sd

plt.style.use('ggplot')

def main(argv):
    if(len(argv) < 2):
        print 'usage :$python process_tranform.py filename'
        sys.exit('No input csv file from command line')

    filename = argv[1]

    if not filename.endswith('csv'):
        print 'Please input a csv file'
        sys.exit('Wrong input file type')
    df = pd.DataFrame.from_csv(filename, index_col = None)

    # remove Null value and merge same string with the same meaning
    df['Title'] = df['Title'].replace({np.NAN:'unknown'})
    df['Title'] = df['Title'].replace({'Ms':'Ms.'})
    print df['Title']



    # random sampling
    print 'Display random sampling'
    print df.sample(5)

    # Ordinal numeric: YearlyIncome ,  TotalChildren,  NumberChildrenAtHome, NumberCarsOwned 
    # Variance and Standard Deviation for ordinal attributes
    ordinal_atts = ['YearlyIncome',  'TotalChildren',  'NumberChildrenAtHome', 'NumberCarsOwned', 'Age']
    ordinal_df = df.loc[:, ordinal_atts]

    # get variance
    print '\nget variance for ordinal attributes\n'
    
    for att in ordinal_atts:
        print att + '     ', ordinal_df.var( axis = 0)[att]
    print '\nget std for ordinal attributes\n'
    # get standard deviation
    print ordinal_df.std(axis = 0)
    del ordinal_df


    # # Binning/Histogram on continuous attributes or categorical attributes (Age, YearlyIncome)
    numeric_atts = ['Age', 'YearlyIncome']
    discrete_df = df.loc[:, numeric_atts ]
    print 'display max and min values for YearlyIncome: '
    print discrete_df['YearlyIncome'].min(), discrete_df['YearlyIncome'].max()

    print 'display max and min values for Age: '

    print discrete_df['Age'].min(), discrete_df['Age'].max()
    print '\n'

    bins = dict()
    bins[0] = [30, 40, 50, 60, 70, 80, 90, 100, 110]
    bins[1] = [10000, 30000, 50000, 70000, 90000, 110000, 130000, 150000, 170000]
    group_names = [['30', '40', '50', '60', '70', '80', '90', '100'],
     ['10000', '30000', '50000', '70000', '90000', '11000', '13000', '15000+']]
    

    #create a subplot fig
    fig, axe = plt.subplots(nrows = 2, ncols = 1)
    idx = 0
    for att in numeric_atts:
        bin_name = att + '_' + 'bins'
        # use cut to create binning values
        discrete_df[bin_name] = pd.cut(discrete_df[att], bins[idx], labels = group_names[idx])

        # display value counts
        print att + ' ' + 'value_counts'
        print pd.value_counts(discrete_df[bin_name])
        bin_df = discrete_df[[att, bin_name]].groupby(bin_name)

        # display bin median
        print att + ' ' + 'Median'
        print bin_df.median()
        bin_df.hist(ax = axe[idx], bins=1)
        idx = idx + 1

    del discrete_df


    plt.savefig('HistogramPlot.png')


    # Normalization ['YearlyIncome']
    num_df = df[ordinal_atts]
    min_max_scaler = sp.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(num_df)
    df_normalized = pd.DataFrame(np_scaled)
    idx = 0
    for att in ordinal_atts:
        print '\nprint before and after Normalization'
        print df[att].head(10), df_normalized[idx].head(10)
        idx += 1
    
    # Standardlization 
    data = df[ordinal_atts].as_matrix()
    scaler = sp.StandardScaler().fit(data)
    np_scaled = scaler.transform(data)    
    print df[ordinal_atts]
    print '\n'
    
    std_data =  dict()
    for idx, att in enumerate(ordinal_atts):
       std_data[att] = np_scaled[:,idx]
    std_df = pd.DataFrame.from_dict(std_data)
    print '\n Standardlization'
    print std_df

    # Binarization (One Hot Encoding)
    categorical_atts = ['Gender', 'Title', 'MaritalStatus',
    'EnglishEducation', 'SpanishEducation', 'SpanishEducation',
    'FrenchEducation', 'EnglishOccupation', 'SpanishOccupation',
    'FrenchOccupation', 'HouseOwnerFlag', 'NumberCarsOwned', 'CommuteDistance','Region']
    
    categorical_df = df[categorical_atts]

    #convert strings values to numerical values, for example: att = ['low', 'medium', 'high']
    #converted to att = [0,1,2]
    le = sp.LabelEncoder()
    categorical_df_1 = categorical_df.apply(le.fit_transform)
    print 'print LabelEncoder'
    print categorical_df_1.head()

    #create One hot encoding
    enc = sp.OneHotEncoder()
    enc_scaler = enc.fit(categorical_df_1)
    onehotlabels =  enc.transform(categorical_df_1).toarray()

    print 'print one hot Encoding result'
    print onehotlabels[:,:10]
   
    # Test Hamming dissimilarity & Jaccard Similarity
    vect_1 = [0, 1, 0, 1, 0, 1, 0, 0, 1]
    vect_2 = [0, 1, 0, 0, 1, 0, 1, 1, 0]

    hamming_dissimilarity = sd.hamming(vect_1, vect_2)
    print 'hamming_dissimilarity score for test data is    ' + str(hamming_dissimilarity)
    print 'Jaccard_SIM score for test data is    ' + str(Jaccard_SIM(vect_1, vect_2)) 
    
    # print results from my transforming data
    vect_1 = onehotlabels[10, :]
    vect_2 = onehotlabels[10000, :]
    hamming_dissimilarity = sd.hamming(vect_1, vect_2)
    print 'hamming_dissimilarity score for my data  is    ' + str(hamming_dissimilarity)
    print 'Jaccard_SIM score for my data is    ' + str(Jaccard_SIM(vect_1, vect_2))

def Jaccard_SIM(vect_1, vect_2):
    total_non_zeros = 0
    total_match = 0
    for i in range(len(vect_1)):
        if vect_1[i] == 1 or vect_2[i] == 1:
            total_non_zeros +=1
            if vect_1[i] == vect_2[i]:
                total_match +=1
    return 1.0*total_match/total_non_zeros


if __name__ == '__main__':
    main(sys.argv)


