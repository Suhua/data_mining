import sys
import pandas as pd
import pdb
def main(argv):
    if len(argv) < 2:
        print 'input the csv file'
        sys.exit('No input file from command line, exit!')
    try: 
        filename = argv[1]
        df = pd.DataFrame.from_csv(filename)
        # list all the attributes
        cols = df.columns.values.tolist()
        print cols

        # select the attributes
        atts = ['GeographyKey','Title','MaritalStatus', 'Gender',
        'YearlyIncome','TotalChildren', 'NumberChildrenAtHome','EnglishEducation',
        'SpanishEducation','FrenchEducation','EnglishOccupation','SpanishOccupation','FrenchOccupation',
        'HouseOwnerFlag','NumberCarsOwned','CommuteDistance', 'Region', 'Age', 'BikeBuyer']

        # create a new dataframe using selected attributes
        data = df.loc[:,atts]
        data.to_csv('vTargetBuyers.csv')

        # release memoery
        del cols, df
        
        # take a look of of the uniqe values in each selected attributes and spot the missing value
        for att in atts:
            print att + ' : '
            print set([x for x in data.loc[:, att].values.tolist()])

    except Exception as e:
        print str(e)
        raise




if __name__ == '__main__':
    main(sys.argv)
